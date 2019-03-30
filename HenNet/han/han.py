import keras
from keras import backend as K
from keras.layers import Dense, GRU, TimeDistributed, Input, Embedding, Bidirectional, Concatenate, Lambda
from keras.models import Model

class AttentionLayer(keras.layers.Layer):
    def __init__(self, context_vector_length=100, **kwargs):
        """
        An implementation of a attention layer. This layer
        accepts a 3d Tensor (batch_size, time_steps, input_dim) and
        applies a single layer attention mechanism in the time
        direction (the second axis).
        :param context_vector_lenght: (int) The size of the hidden context vector.
            If set to 1 this layer reduces to a standard attention layer.
        :param kwargs: Any argument that the baseclass Layer accepts.
        """
        self.context_vector_length = context_vector_length
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[2]

        # Add a weights layer for the
        self.W = self.add_weight(name='W', shape=(dim, self.context_vector_length), initializer=keras.initializers.get('uniform'), trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(self.context_vector_length, 1),initializer=keras.initializers.get('uniform'), trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def _get_attention_weights(self, X):
        """
        Computes the attention weights for each timestep in X
        :param X: 3d-tensor (batch_size, time_steps, input_dim)
        :return: 2d-tensor (batch_size, time_steps) of attention weights
        """
        # Compute a time-wise stimulus, i.e. a stimulus for each
        # time step. For this first compute a hidden layer of
        # dimension self.context_vector_length and take the
        # similarity of this layer with self.u as the stimulus
        u_tw = K.tanh(K.dot(X, self.W))
        tw_stimulus = K.dot(u_tw, self.u)

        # Remove the last axis an apply softmax to the stimulus to
        # get a probability.
        tw_stimulus = K.reshape(tw_stimulus, (-1, tw_stimulus.shape[1]))
        att_weights = K.softmax(tw_stimulus)
        return att_weights

    def call(self, X):
        att_weights = self._get_attention_weights(X)
        # Reshape the attention weights to match the dimensions of X
        att_weights = K.reshape(att_weights, (-1, att_weights.shape[1], 1))
        att_weights = K.repeat_elements(att_weights, X.shape[-1], -1)
        # Multiply each input by its attention weights
        weighted_input = keras.layers.Multiply()([X, att_weights])
        # Sum in the direction of the time-axis.
        weighted_sum = K.sum(weighted_input, axis=1)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def get_config(self):
        config = {'context_vector_length': self.context_vector_length}
        base_config = super(AttentionLayer, self).get_config()
        return {**base_config, **config}

class HAN(Model):
    def __init__(
            self, max_words, max_sentences, output_size,
            embedding_matrix, word_encoding_dim=200,
            sentence_encoding_dim=200, inputs=None,
            outputs=None, name='han-for-docla'):
        self.max_words = max_words
        self.max_sentences = max_sentences
        self.output_size = output_size
        self.embedding_matrix = embedding_matrix
        self.word_encoding_dim = word_encoding_dim
        self.sentence_encoding_dim = sentence_encoding_dim

        in_tensor, out_tensor = self.build_network_default()
        super(HAN, self).__init__(inputs=in_tensor, outputs=out_tensor, name=name)

    def build_word_encoder(self, max_words, embedding_matrix, encoding_dim=200):
        vocabulary_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        embedding_layer = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix], input_length=max_words, trainable=False)
        sentence_input = Input(shape=(max_words,), dtype='int32')
        embedded_sentences = embedding_layer(sentence_input)
        encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(embedded_sentences)
        return Model(inputs=[sentence_input], outputs=[encoded_sentences], name='word_encoder')

    def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim=200):
        text_input = Input(shape=(max_sentences, summary_dim))
        encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(text_input)
        return Model(inputs=[text_input], outputs=[encoded_sentences], name='sentence_encoder')

    def build_network_default(self):
        in_tensor = Input(shape=(self.max_sentences, self.max_words))
        word_encoder = self.build_word_encoder(self.max_words, self.embedding_matrix, self.word_encoding_dim)
        word_rep = TimeDistributed(word_encoder, name='word_encoder')(in_tensor)

        # Sentence Rep is a 3d-tensor (batch_size, max_sentences, word_encoding_dim)
        sentence_rep = TimeDistributed(AttentionLayer(), name='word_attention')(word_rep)
        doc_rep = self.build_sentence_encoder(self.max_sentences, self.word_encoding_dim, self.sentence_encoding_dim)(sentence_rep)

        doc_summary = AttentionLayer(name='sentence_attention')(doc_rep)
        out_tensor = Dense(self.output_size, activation='softmax', name='class_prediction')(doc_summary)
        return in_tensor, out_tensor

    def get_config(self):
        config = {
            'max_words': self.max_words,
            'max_sentences': self.max_sentences,
            'output_size': self.output_size,
            'embedding_matrix': self.embedding_matrix,
            'word_encoding_dim': self.word_encoding_dim,
            'sentence_encoding_dim': self.sentence_encoding_dim,
            'base_config': super(HAN, self).get_config()}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        base_config = config.pop('base_config')
        return Model.from_config(base_config, custom_objects=custom_objects)

    def predict_sentence_attention(self, X):
        """
        For a given set of texts predict the attention
        weights for each sentence.
        :param X: 3d-tensor, similar to the input for predict
        :return: 2d array (num_obs, max_sentences) containing
            the attention weights for each sentence
        """
        att_layer = self.get_layer('sentence_attention')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = Lambda(lambda x: att_layer._get_attention_weights(x))(prev_tensor)
        return Model(self.input, dummy_layer).predict(X)