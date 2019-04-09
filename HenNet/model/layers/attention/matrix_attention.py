from keras import backend as K
from keras import initializers
from overrides import overrides
from model.layers import MaskedLayer

class MatrixAttention(MaskedLayer):
    '''
    This ``Layer`` takes two matrices as input and returns a matrix of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.  We don't worry about zeroing out any masked values, because we propagate a correct
    mask.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.

    This is largely similar to using ``TimeDistributed(Attention)``, except the result is
    unnormalized, and we return a mask, so you can do a masked normalization with the result.  You
    should use this instead of ``TimeDistributed(Attention)`` if you want to compute multiple
    normalizations of the attention matrix.

    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim)``, with mask
          ``(batch_size, num_rows_1)``
        - matrix_2: ``(batch_size, num_rows_2, embedding_dim)``, with mask
          ``(batch_size, num_rows_2)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``, with mask of same shape

    Parameters
    ----------
    '''
    def __init__(self, similarity_function='dot', **kwargs):
        self.similarity_function = similarity_function
        self.init = initializers.get('glorot_uniform')
        super(MatrixAttention, self).__init__(**kwargs)

    @overrides
    def build(self, input_shape):
        tensor_1_dim = input_shape[0][-1]
        tensor_2_dim = input_shape[1][-1]

        if self.similarity_function == 'dot':
            assert tensor_1_dim == tensor_2_dim
            self.trainable_weights = []

        elif self.similarity_function == 'bilinear':
            self.weight_matrix = K.variable(self.init((tensor_1_dim, tensor_2_dim)), name=self.name + "_weights")
            self.bias = K.variable(self.init((1,)), name=self.name + "_bias")
            self.trainable_weights = [self.weight_matrix, self.bias]

        super(MatrixAttention, self).build(input_shape)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        mask_1, mask_2 = mask
        if mask_1 is None and mask_2 is None:
            return None
        if mask_1 is None:
            mask_1 = K.ones_like(K.sum(inputs[0], axis=-1))
        if mask_2 is None:
            mask_2 = K.ones_like(K.sum(inputs[1], axis=-1))
        mask_1 = K.cast(K.expand_dims(mask_1, axis=2), 'float32')
        mask_2 = K.cast(K.expand_dims(mask_2, axis=1), 'float32')
        return K.cast(K.batch_dot(mask_1, mask_2), 'uint8')

    @overrides
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])

    @overrides
    def call(self, inputs, mask=None):
        matrix_1, matrix_2 = inputs
        num_rows_1 = K.shape(matrix_1)[1]
        num_rows_2 = K.shape(matrix_2)[1]
        tile_dims_1 = K.concatenate([[1, 1], [num_rows_2], [1]], 0)
        tile_dims_2 = K.concatenate([[1], [num_rows_1], [1, 1]], 0)
        tiled_matrix_1 = K.tile(K.expand_dims(matrix_1, axis=2), tile_dims_1)
        tiled_matrix_2 = K.tile(K.expand_dims(matrix_2, axis=1), tile_dims_2)

        if self.similarity_function == 'dot':
            similarity = K.sum(tiled_matrix_1 * tiled_matrix_2, axis=-1)
        elif self.similarity_function == 'bilinear':
            dot_product = K.sum(K.dot(tiled_matrix_1, self.weight_matrix) * tiled_matrix_2, axis=-1)
            similarity = K.tanh(dot_product + self.bias)
        return similarity

    @overrides
    def get_config(self):
        base_config = super(MatrixAttention, self).get_config()
        config = {'similarity_function': self.similarity_function}
        config.update(base_config)
        return config