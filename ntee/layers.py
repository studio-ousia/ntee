# -*- coding: utf-8 -*-

import numpy as np
import theano
from keras import backend as K
from keras.backend.common import floatx
from keras.engine.topology import Layer
from theano import tensor as T

FLOATX = floatx()
FLOAT_MIN = np.finfo('float32').min + K.epsilon()
FLOAT_MAX = np.finfo('float32').max - K.epsilon()


class TextRepresentationLayer(Layer):
    def __init__(self, W=None, b=None, *args, **kwargs):
        super(TextRepresentationLayer, self).__init__(*args, **kwargs)

        self.W = W
        self.b = b

    def build(self, input_shape):
        if self.W is None:
            self.W = K.variable(np.identity(input_shape[0][2]))
        elif isinstance(self.W, np.ndarray):
            self.W = K.variable(self.W)
        else:
            raise RuntimeError()

        if self.b is None:
            self.b = K.random_uniform_variable((input_shape[0][2],), -0.05, 0.05)
        elif isinstance(self.b, np.ndarray):
            self.b = K.variable(self.b)
        else:
            raise RuntimeError()

        self.trainable_weights = [self.W, self.b]

    def call(self, inputs, mask=None):
        def f(i, embedding, text_input):
            mask = T.neq(text_input[i], 0).astype(FLOATX)
            vec = T.dot(mask, embedding[i])
            vec /= T.maximum(vec.norm(2, 0), K.epsilon())

            return T.dot(vec, self.W) + self.b

        return theano.map(f, T.arange(inputs[0].shape[0]), non_sequences=inputs)[0]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])


class DotLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(DotLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        l1 = inputs[0]
        l2 = inputs[1]

        def f(i, l1, l2):
            return T.clip(T.batched_tensordot(l1[i], l2[i], 1), FLOAT_MIN, FLOAT_MAX).astype(FLOATX)

        return theano.map(f, T.arange(l1.shape[0]), non_sequences=[l1, l2])[0]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])


# We use this class instead of the Theano's softmax to avoid this reported issue:
# https://github.com/Theano/Theano/issues/3162
class SoftmaxLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(SoftmaxLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        inputs_exp = T.exp(inputs)
        denominator = inputs_exp.sum(1, keepdims=True)
        return inputs_exp / T.clip(denominator, K.epsilon(), FLOAT_MAX).astype(FLOATX)

    def get_output_shape_for(self, input_shape):
        return input_shape
