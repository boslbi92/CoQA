import keras
from keras import backend as K
from keras.layers import Dense, GRU, TimeDistributed, Input, Embedding, Bidirectional, Concatenate, Lambda
from keras.models import Model

class AttentionLayer(keras.layers.Layer):
    def __init__(self, context_vector_length=100, **kwargs):
        self.context_vector_length = context_vector_length
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[2]

        # Add a weights layer for the
        self.W = self.add_weight(name='W', shape=(dim, self.context_vector_length), initializer=keras.initializers.get('uniform'), trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(self.context_vector_length, 1),initializer=keras.initializers.get('uniform'), trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def _get_attention_weights(self, X):
        u_tw = K.tanh(K.dot(X, self.W))
        tw_stimulus = K.dot(u_tw, self.u)
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

class MatrixAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MatrixAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []
        super(MatrixAttention, self).build(input_shape)

    def call(self, X):
        matrix_1, matrix_2 = X
        print (matrix_1)
        axis1, axis2, axis3 = matrix_1.shape[1], matrix_1.shape[2], matrix_1.shape[3]
        matrix_1_reshaped, matrix_2_reshaped = (K.reshape(matrix_1, shape=((axis1*axis2), axis3))), (K.reshape(matrix_2, shape=((axis1*axis2), axis3)))
        print (matrix_1_reshaped, matrix_2_reshaped)
        similarity = (K.dot(matrix_1_reshaped, K.transpose(matrix_2_reshaped)))
        print (similarity)
        print (K.reshape(similarity, shape=(axis1, axis2, axis3)))

        num_rows_1 = K.shape(matrix_1)[1]
        num_rows_2 = K.shape(matrix_2)[1]
        tile_dims_1 = K.concatenate([[1, 1], [num_rows_2], [1]], 0)
        tile_dims_2 = K.concatenate([[1], [num_rows_1], [1, 1]], 0)
        tiled_matrix_1 = K.tile(K.expand_dims(matrix_1, axis=2), tile_dims_1)
        tiled_matrix_2 = K.tile(K.expand_dims(matrix_2, axis=1), tile_dims_2)
        similarity = K.sum(tiled_matrix_1 * tiled_matrix_2, axis=-1)
        return similarity

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])

    def get_config(self):
        base_config = super(MatrixAttention, self).get_config()
        config.update(base_config)
        return config