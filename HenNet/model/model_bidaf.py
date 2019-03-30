import numpy as np
from overrides import overrides
from keras import backend as K
from keras import Model, optimizers
from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed, Reshape
from keras.layers import LSTM, GRU, Bidirectional, Dropout
from model.layers.attention import MatrixAttention, WeightedSum, MaskedSoftmax
from model.layers.backend import Max, Repeat, RepeatLike, StackProbs
from model.metrics.custom_metrics import monitor_span, negative_log_span
# from HenNet.model.layers import Highway
import os, time

class BiDAF():
    def __init__(self):
        self.num_hidden_seq2seq_layers = 2
        self.embedding_dim = 300
        self.vocab_size = 30000
        self.num_passage_words = 500
        self.num_question_words = 20
        self.num_highway_layers = 2
        self.highway_activation = 'relu'

    def build_model(self, context_input, history_input, output, epochs=5):
        # PART 1: First we create input layers
        question_input = Input(shape=(self.num_question_words, self.embedding_dim), dtype='float32', name="question_input")
        passage_input = Input(shape=(self.num_passage_words, self.embedding_dim), dtype='float32', name="passage_input")

        # PART 2: Build encoders
        encoding_dim = int(self.embedding_dim/2)
        encoded_question = Bidirectional(GRU(encoding_dim, return_sequences=True), name='encoded_question')(question_input) # Shape: (batch_size, #words, embedding_dim)
        encoded_passage = Bidirectional(GRU(encoding_dim, return_sequences=True), name='encoded_passage')(passage_input) # Shape: (batch_size, #words, embedding_dim)

        # PART 3: Now we compute a similarity between the passage words and the question words
        # Shape: (batch_size, num_passage_words, num_question_words)
        matrix_attention = MatrixAttention(similarity_function='dot')([encoded_passage, encoded_question])

        # PART 3-1: Context-to-query (c2q) attention (normalized over question)
        # Shape: (batch_size, num_passage_words, embedding_dim)
        passage_question_attention = MaskedSoftmax()(matrix_attention)
        c2q_vectors = WeightedSum(name="c2q_vectors", use_masking=False)([encoded_question, passage_question_attention])

        # PART 3-2: Query-to-context (q2c) attention (normalized over context)
        # For each document word, the most similar question word to it, and computes a single attention over the whole document using these max similarities.
        # Shape: (batch_size, num_passage_words)
        question_passage_similarity = Max(axis=-1)(matrix_attention) # Shape: (batch_size, num_passage_words)
        question_passage_attention = MaskedSoftmax()(question_passage_similarity) # Shape: (batch_size, num_passage_words)
        q2c_vectors = WeightedSum(name="q2c_vectors", use_masking=False)([encoded_passage, question_passage_attention])

        # Repeats question/passage vector for every word in the passage, and uses as an additional input to the hidden layers above.
        tiled_q2c_vectors = RepeatLike(axis=1, copy_from_axis=1)([q2c_vectors, encoded_passage]) # Shape: (batch_size, num_passage_words, embedding_dim)
        merged = Concatenate(name='attention_output')([encoded_passage, c2q_vectors, tiled_q2c_vectors]) # Shape: (batch_size, num_passage_words, embedding_dim * 3)

        # PART 4: Final modelling layer 
        # Having computed a combined representation of the document that includes attended question
        # vectors, pass this through a few more bi-directional encoder layers, then predict the span_begin word.
        modeled_passage = Bidirectional(GRU(encoding_dim, return_sequences=True), name='modelling_encoder')(merged)

        # To predict the span word, we pass the merged representation through a Dense layer without
        # output size 1 (basically a dot product of a vector of weights and the passage vectors),
        # then do a softmax to get a position.
        span_begin_input = Concatenate(name='final_representation')([merged, modeled_passage])
        span_begin_weights = TimeDistributed(Dense(units=1), name='span_begin_weights')(span_begin_input)
        span_begin_probabilities = MaskedSoftmax(name="span_begin_probs")(span_begin_weights) # Shape: (batch_size, num_passage_words)

        # PART 4:
        # Given what we predicted for span_begin, we'll pass it through a final encoder layer and
        # predict span_end.  NOTE: I'm following what Min did in his _code_, not what it says he
        # did in his _paper_.  The equations in his paper do not mention that he did this last
        # weighted passage representation and concatenation before doing the final biLSTM (though
        # his figure makes it clear this is what he intended; he just wrote the equations wrong).
        # Shape: (batch_size, num_passage_words, embedding_dim * 2)
        sum_layer = WeightedSum(name="passage_weighted_by_predicted_span", use_masking=False)
        repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
        passage_weighted_by_predicted_span = repeat_layer([sum_layer([modeled_passage, span_begin_probabilities]), encoded_passage])

        span_end_representation = Concatenate()([merged, modeled_passage, passage_weighted_by_predicted_span])
        final_seq2seq = Bidirectional(GRU(encoding_dim, return_sequences=True), name='span_end_encoder')(span_end_representation)
        span_end_input = Concatenate()([merged, final_seq2seq])
        span_end_weights = TimeDistributed(Dense(units=1), name='span_end_weights')(span_end_input)
        span_end_probabilities = MaskedSoftmax(name="span_end_probs")(span_end_weights)

        prob_output = StackProbs()([span_begin_probabilities, span_end_probabilities])

        bidaf = Model(inputs=[question_input, passage_input], outputs=[prob_output])
        bidaf.compile(optimizer='adadelta', loss=negative_log_span)
        time.sleep(1.0)
        bidaf.summary(line_length=175)
        bidaf.fit(x=[history_input, context_input], y=[output], epochs=epochs, batch_size=20,
                  shuffle=True, validation_split=0.2, callbacks=[monitor_span()])

    def _get_custom_objects(self, cls):
        custom_objects = super(BidirectionalAttentionFlow, cls)._get_custom_objects()
        custom_objects["ComplexConcat"] = ComplexConcat
        custom_objects["MaskedSoftmax"] = MaskedSoftmax
        custom_objects["MatrixAttention"] = MatrixAttention
        custom_objects["Max"] = Max
        custom_objects["Repeat"] = Repeat
        custom_objects["RepeatLike"] = RepeatLike
        custom_objects["WeightedSum"] = WeightedSum
        return custom_objects

