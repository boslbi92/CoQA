import numpy as np
from overrides import overrides
from keras import backend as K
from keras import Model, optimizers
from keras.regularizers import l2
from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed, Reshape
from keras.layers import LSTM, GRU, Bidirectional, Dropout, Add, CuDNNGRU
from keras.optimizers import RMSprop, Adamax
from model.layers.attention import MatrixAttention, WeightedSum, MaskedSoftmax
from model.layers.backend import Max, Repeat, RepeatLike, ComplexConcat, StackProbs
from model.metrics.custom_metrics import monitor_span, negative_log_span
import os, time

class HenNet_GPU():
    def __init__(self, c_pad, h_pad, hidden_dim):
        self.embedding_dim = 1113
        self.encoding_dim = int(hidden_dim / 2)
        self.num_passage_words = c_pad
        self.num_question_words = h_pad
        self.dropout_rate = 0.3

    def build_model(self):
        encoding_dim = self.encoding_dim

        # PART 1: First we create input layers
        question_input = Input(shape=(self.num_question_words, self.embedding_dim), dtype='float32', name="question_input")
        passage_input = Input(shape=(self.num_passage_words, self.embedding_dim), dtype='float32', name="passage_input")

        # PART 2: Build encoders
        encoded_question_1 = Bidirectional(CuDNNGRU(encoding_dim, return_sequences=True), name='question_encoder_1')(question_input)
        encoded_question_2 = Bidirectional(CuDNNGRU(encoding_dim, return_sequences=True), name='question_encoder_2')(encoded_question_1)
        encoded_question = Add(name='sum_question_encoder')([encoded_question_1, encoded_question_2])

        encoded_passage_1 = Bidirectional(CuDNNGRU(encoding_dim, return_sequences=True), name='passage_encoder1')(passage_input)
        encoded_passage_2 = Bidirectional(CuDNNGRU(encoding_dim, return_sequences=True), name='passage_encoder2')(encoded_passage_1)
        encoded_passage = Add(name='sum_passage_encoder')([encoded_passage_1, encoded_passage_2])

        # PART 3: Now we compute a similarity between the passage words and the question words
        matrix_attention = MatrixAttention(similarity_function='bilinear', name='similarity_matrix')([encoded_passage, encoded_question])

        # PART 3-1: Context-to-query (c2q) attention (normalized over question)
        passage_question_attention = MaskedSoftmax(name='normalize_c2q')(matrix_attention)
        c2q_vectors = WeightedSum(name="c2q_attention", use_masking=False)([encoded_question, passage_question_attention])

        # PART 3-2: Query-to-context (q2c) attention (normalized over context)
        # For each document word, the most similar question word to it, and computes a single attention over the whole document using these max similarities.
        question_passage_similarity = Max(axis=-1, name='maxpool_col')(matrix_attention) # Shape: (batch_size, num_passage_words)
        question_passage_attention = MaskedSoftmax(name='normalize_q2c')(question_passage_similarity) # Shape: (batch_size, num_passage_words)
        q2c_vectors = WeightedSum(use_masking=False, name='attended_context_vector')([encoded_passage, question_passage_attention])

        # PART 3-3: Final attention output
        # Repeats question/passage vector for every word in the passage, and uses as an additional input to the hidden layers above.
        tiled_q2c_vectors = RepeatLike(axis=1, copy_from_axis=1, name="q2c_attention")([q2c_vectors, encoded_passage])
        attention_output = ComplexConcat(combination='1,2,1*2,1*3', name='attention_output')([encoded_passage, c2q_vectors, tiled_q2c_vectors])

        # PART 4: Final modelling layer
        final_encoder1 = Bidirectional(CuDNNGRU(encoding_dim, return_sequences=True), name='final_encoder1')(attention_output)
        final_encoder2 = Bidirectional(CuDNNGRU(encoding_dim, return_sequences=True), name='final_encoder2')(final_encoder1)
        output_representation = Concatenate(name='output_representation')([attention_output, final_encoder2])
        output_representation = Dropout(rate=self.dropout_rate, name='output_rep_drop')(output_representation)

        # PART 5-1: Span prediction layers (begin)
        # To predict the span word, we pass the output representation through each dense layers without
        # output size 1 (basically a dot product of a vector of weights and the output vectors) + softmax (to get a position)
        span_begin_weights = TimeDistributed(Dense(units=1, activation='tanh'), name='span_begin_weights')(output_representation)
        span_begin_probabilities = MaskedSoftmax(name="output_begin_probs")(span_begin_weights)

        # PART 5-1: Weighted passages by span begin probs
        sum_layer = WeightedSum(name="weighted_passages", use_masking=False)
        repeat_layer = RepeatLike(axis=1, copy_from_axis=1, name='tiled_weighted_passages')
        weighted_passages = repeat_layer([sum_layer([final_encoder2, span_begin_probabilities]), encoded_passage])
        span_end_representation = ComplexConcat(combination="1,2,3,2*3")([attention_output, final_encoder2, weighted_passages])

        # PART 5-2: Span prediction layers (end)
        span_end_encoder = Bidirectional(CuDNNGRU(int(encoding_dim/2), return_sequences=True), name='span_end_encoder')(span_end_representation)
        span_end_input = Concatenate(name='span_end_representation')([attention_output, span_end_encoder])
        span_end_input = Dropout(rate=self.dropout_rate, name='span_end_rep_drop')(span_end_input)
        span_end_weights = TimeDistributed(Dense(units=1, activation='tanh'), name='span_end_weights')(span_end_input)
        span_end_probabilities = MaskedSoftmax(name="output_end_probs")(span_end_weights)

        prob_output = StackProbs(name='final_span_outputs')([span_begin_probabilities, span_end_probabilities])

        # Model hyperparams
        opt = Adamax(clipvalue=5.0, lr=0.002)
        henNet = Model(inputs=[question_input, passage_input], outputs=[prob_output])
        henNet.compile(optimizer=opt, loss=negative_log_span)
        time.sleep(1.0)
        henNet.summary(line_length=175)
        return henNet


    def _get_custom_objects(self):
        custom_objects = super(HenNet_GPU, self)._get_custom_objects()
        custom_objects["ComplexConcat"] = ComplexConcat
        custom_objects["MaskedSoftmax"] = MaskedSoftmax
        custom_objects["MatrixAttention"] = MatrixAttention
        custom_objects["Max"] = Max
        custom_objects["Repeat"] = Repeat
        custom_objects["RepeatLike"] = RepeatLike
        custom_objects["WeightedSum"] = WeightedSum
        custom_objects["StackProbs"] = StackProbs
        return custom_objects

