# -*- coding: utf-8 -*-

from keras.layers import Input, RepeatVector
from keras.layers.embeddings import Embedding
from keras.models import Model

from layers import TextRepresentationLayer, DotLayer, SoftmaxLayer


def build_model(text_len, negative_size, optimizer, word_size, entity_size,
                dim_size, word_static, entity_static, word_embedding, entity_embedding):
    text_input_layer = Input(shape=(text_len,), dtype='int32')

    word_embed_layer = Embedding(
        word_size, dim_size, input_length=text_len, name='word_embedding',
        weights=[word_embedding], trainable=not word_static
    )(text_input_layer)

    text_layer = TextRepresentationLayer(name='text_layer')(
        [word_embed_layer, text_input_layer]
    )

    entity_input_layer = Input(shape=(negative_size + 1,), dtype='int32')

    entity_embed_layer = Embedding(
        entity_size, dim_size, input_length=negative_size + 1,
        name='entity_embedding', weights=[entity_embedding],
        trainable=not entity_static
    )(entity_input_layer)

    similarity_layer = DotLayer(name='dot_layer')(
        [RepeatVector(negative_size + 1)(text_layer), entity_embed_layer]
    )

    predictions = SoftmaxLayer()(similarity_layer)

    model = Model(input=[text_input_layer, entity_input_layer],
                  output=predictions)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
