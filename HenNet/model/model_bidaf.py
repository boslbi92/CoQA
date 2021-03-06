import numpy as np
from overrides import overrides
from keras import backend as K
from keras import Model, optimizers
from keras.regularizers import l2
from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed, Reshape
from keras.layers import LSTM, GRU, Bidirectional, Dropout
from model.layers.attention import MatrixAttention, WeightedSum, MaskedSoftmax
from model.layers.backend import Max, Repeat, RepeatLike, ComplexConcat, StackProbs
from model.metrics.custom_metrics import monitor_span, negative_log_span
from keras.callbacks import TensorBoard, ModelCheckpoint
import os, time

class BiDAF():
    def __init__(self):
        self.embedding_dim = 300
        self.num_passage_words = 1200
        self.num_question_words = 30
        self.dropout_rate = 0.2
        self.regularizer = l2(l=0.001)
        self.tensorboard = TensorBoard(log_dir='train_logs/{}'.format(time.time()))
        self.model_dir = os.getcwd() + '/saved_models/{epoch:02d}-{val_loss:.3f}.hdf5'
        self.checkpoint = ModelCheckpoint(self.model_dir, monitor='val_loss')

    def build_model(self, context_input, history_input, output, epochs=5):
        # PART 1: First we create input layers
        question_input = Input(shape=(self.num_question_words, self.embedding_dim), dtype='float32', name="question_input")
        passage_input = Input(shape=(self.num_passage_words, self.embedding_dim), dtype='float32', name="passage_input")

        # PART 2: Build encoders
        # Shape: (batch_size, #words, embedding_dim)
        encoding_dim = int(self.embedding_dim/2)
        encoded_question = Bidirectional(GRU(encoding_dim, return_sequences=True, dropout=self.dropout_rate), name='question_encoder')(question_input)
        encoded_passage = Bidirectional(GRU(encoding_dim, return_sequences=True, dropout=self.dropout_rate), name='passage_encoder')(passage_input)

        # PART 3: Now we compute a similarity between the passage words and the question words
        # Shape: (batch_size, num_passage_words, num_question_words)
        matrix_attention = MatrixAttention(similarity_function='dot', name='similarity_matrix')([encoded_passage, encoded_question])

        # PART 3-1: Context-to-query (c2q) attention (normalized over question)
        # Shape: (batch_size, num_passage_words, embedding_dim)
        passage_question_attention = MaskedSoftmax(name='normalize_c2q')(matrix_attention)
        c2q_vectors = WeightedSum(name="c2q_attention", use_masking=False)([encoded_question, passage_question_attention])

        # PART 3-2: Query-to-context (q2c) attention (normalized over context)
        # For each document word, the most similar question word to it, and computes a single attention over the whole document using these max similarities.
        # Shape: (batch_size, num_passage_words)
        question_passage_similarity = Max(axis=-1, name='maxpool_col')(matrix_attention) # Shape: (batch_size, num_passage_words)
        question_passage_attention = MaskedSoftmax(name='normalize_q2c')(question_passage_similarity) # Shape: (batch_size, num_passage_words)
        q2c_vectors = WeightedSum(use_masking=False, name='attended_context_vector')([encoded_passage, question_passage_attention])

        # PART 3-3: Final attention output
        # Repeats question/passage vector for every word in the passage, and uses as an additional input to the hidden layers above.
        # Shape: (batch_size, num_passage_words, embedding_dim * 4)
        tiled_q2c_vectors = RepeatLike(axis=1, copy_from_axis=1, name="q2c_attention")([q2c_vectors, encoded_passage])
        attention_output = ComplexConcat(combination='1,2,1*2,1*3', name='attention_output')([encoded_passage, c2q_vectors, tiled_q2c_vectors])

        # PART 4: Final modelling layer
        final_encoder1 = Bidirectional(GRU(encoding_dim, return_sequences=True, dropout=self.dropout_rate), name='final_encoder1')(attention_output)
        final_encoder2 = Bidirectional(GRU(encoding_dim, return_sequences=True, dropout=self.dropout_rate), name='final_encoder2')(final_encoder1)
        output_representation = Concatenate(name='output_representation')([attention_output, final_encoder2])

        # PART 5-1: Span prediction layers (begin)
        # To predict the span word, we pass the output representation through each dense layers without
        # output size 1 (basically a dot product of a vector of weights and the output vectors) + softmax (to get a position)
        # Shape: (batch_size, num_passage_words)
        span_begin_weights = TimeDistributed(Dense(units=1, kernel_regularizer=self.regularizer), name='span_begin_weights')(output_representation)
        span_begin_probabilities = MaskedSoftmax(name="output_begin_probs")(span_begin_weights)

        # PART 5-1: Weighted passages by span begin probs
        # Given what we predicted for span_begin, we'll pass it through a final encoder layer and
        # predict span_end.  NOTE: I'm following what Min did in his _code_, not what it says he
        # did in his _paper_. The equations in his paper do not mention that he did this last
        # weighted passage representation and concatenation before doing the final LSTM
        sum_layer = WeightedSum(name="weighted_passages", use_masking=False)
        repeat_layer = RepeatLike(axis=1, copy_from_axis=1, name='tiled_weighted_passages')
        weighted_passages = repeat_layer([sum_layer([final_encoder2, span_begin_probabilities]), encoded_passage])
        span_end_representation = Concatenate()([attention_output, final_encoder2, weighted_passages])

        # PART 5-2: Span prediction layers (end)
        span_end_encoder = Bidirectional(GRU(encoding_dim, return_sequences=True, dropout=self.dropout_rate), name='span_end_encoder')(span_end_representation)
        span_end_input = Concatenate(name='span_end_representation')([attention_output, span_end_encoder])
        span_end_weights = TimeDistributed(Dense(units=1, kernel_regularizer=self.regularizer), name='span_end_weights')(span_end_input)
        span_end_probabilities = MaskedSoftmax(name="output_end_probs")(span_end_weights)
        prob_output = StackProbs(name='final_span_outputs')([span_begin_probabilities, span_end_probabilities])

        # Model hyperparams
        bidaf = Model(inputs=[question_input, passage_input], outputs=[prob_output])
        bidaf.compile(optimizer='adadelta', loss=negative_log_span)
        time.sleep(1.0)
        bidaf.summary(line_length=175)
        bidaf.fit(x=[history_input, context_input], y=[output], epochs=epochs, batch_size=20,
                  shuffle=True, validation_split=0.2, callbacks=[monitor_span(), self.tensorboard, self.checkpoint])

    def _get_custom_objects(self):
        custom_objects = super(BiDAF, self)._get_custom_objects()
        custom_objects["ComplexConcat"] = ComplexConcat
        custom_objects["MaskedSoftmax"] = MaskedSoftmax
        custom_objects["MatrixAttention"] = MatrixAttention
        custom_objects["Max"] = Max
        custom_objects["Repeat"] = Repeat
        custom_objects["RepeatLike"] = RepeatLike
        custom_objects["WeightedSum"] = WeightedSum
        custom_objects["StackProbs"] = StackProbs
        return custom_objects

