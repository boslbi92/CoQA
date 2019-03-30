from keras import backend as K
from overrides import overrides
from model.layers import MaskedLayer

class StackProbs(MaskedLayer):
    def __init__(self, **kwargs):
        super(StackProbs, self).__init__(**kwargs)

    @overrides
    def build(self, input_shape):
        self.trainable_weights = []
        super(StackProbs, self).build(input_shape)

    @overrides
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], len(input_shape), input_shape[0][1])

    @overrides
    def call(self, inputs, mask=None):
        start, end = inputs[0], inputs[1]
        start, end = K.expand_dims(start, axis=1), K.expand_dims(end, axis=1)
        stacked_prob = K.concatenate([start, end], axis=1)
        return stacked_prob

    @overrides
    def get_config(self):
        base_config = super(StackProbs, self).get_config()
        config.update(base_config)
        return config
