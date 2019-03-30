import keras
from keras import backend as K
from keras.layers import Dense, GRU, TimeDistributed, Input, Embedding, Bidirectional, Concatenate, Lambda
from keras.models import Model
from Attention import AttentionLayer, MatrixAttention

class HenNet(Model):
    def __init__(self, max_words, max_sentences, output_size, embedding_matrix, word_encoding_dim=200,
                 sentence_encoding_dim=200, inputs=None, outputs=None, name='han-for-docla'):
        self.max_words = max_words
        self.max_sentences = max_sentences
        self.output_size = output_size
        self.embedding_matrix = embedding_matrix
        self.word_encoding_dim = word_encoding_dim
        self.sentence_encoding_dim = sentence_encoding_dim

        in_tensor, out_tensor = self.build_network_bidaf()
        super(HenNet, self).__init__(inputs=in_tensor, outputs=out_tensor, name=name)

    def build_word_encoder(self, max_words, embedding_matrix, encoding_dim=200):
        vocabulary_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        embedding_layer = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix], input_length=max_words, trainable=False)
        sentence_input = Input(shape=(max_words,), dtype='int32')
        embedded_sentences = embedding_layer(sentence_input)
        encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(embedded_sentences)
        return Model(inputs=[sentence_input], outputs=[encoded_sentences])

    def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim=200):
        text_input = Input(shape=(max_sentences, summary_dim))
        encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(text_input)
        return Model(inputs=[text_input], outputs=[encoded_sentences])

    def build_network_bidaf(self):
        context_input = Input(shape=(self.max_sentences, self.max_words), name='context_input')
        context_word_encoder = self.build_word_encoder(self.max_words, self.embedding_matrix, self.word_encoding_dim)
        context_word_rep = TimeDistributed(context_word_encoder, name='context_word_rnn')(context_input)

        history_input = Input(shape=(self.max_sentences, self.max_words), name='history_input')
        history_word_encoder = self.build_word_encoder(self.max_words, self.embedding_matrix, self.word_encoding_dim)
        history_word_rep = TimeDistributed(history_word_encoder, name='history_word_rnn')(history_input)


        matrix_attention = MatrixAttention()([context_word_rep, history_word_rep])


        # Sentence Rep is a 3d-tensor (batch_size, max_sentences, word_encoding_dim)
        context_sentence_rep = TimeDistributed(AttentionLayer(), name='context_word_attention')(context_word_rep)
        context_rep = self.build_sentence_encoder(self.max_sentences, self.word_encoding_dim, self.sentence_encoding_dim)(context_sentence_rep)
        doc_summary = AttentionLayer(name='context_sentence_attention')(context_rep)

        history_sentence_rep = TimeDistributed(AttentionLayer(), name='history_word_attention')(history_word_rep)
        history_rep = self.build_sentence_encoder(self.max_sentences, self.word_encoding_dim, self.sentence_encoding_dim)(history_sentence_rep)
        history_summary = AttentionLayer(name='history_sentence_attention')(history_rep)

        doc_out = Dense(20, activation='softmax')(doc_summary)
        history_out = Dense(20, activation='softmax')(history_summary)
        concat = Concatenate()([doc_out, history_out])
        pred = Dense(2, activation='softmax', name='final_pred')(concat)
        return ([context_input, history_input], pred)

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