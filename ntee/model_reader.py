# -*- coding: utf-8 -*-

import joblib
import numpy as np

from utils.tokenizer import RegexpTokenizer


class ModelReader(object):
    def __init__(self, model_file):
        model = joblib.load(model_file, mmap_mode='r')
        self._word_embedding = model['word_embedding']
        self._entity_embedding = model['entity_embedding']
        self._W = model.get('W')
        self._b = model.get('b')
        self._vocab = model.get('vocab')

        self._tokenizer = RegexpTokenizer()

    @property
    def vocab(self):
        return self._vocab

    @property
    def word_embedding(self):
        return self._word_embedding

    @property
    def entity_embedding(self):
        return self._entity_embedding

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    def get_word_vector(self, word, default=None):
        index = self._vocab.get_word_index(word)
        if index:
            return self.word_embedding[index]
        else:
            return default

    def get_entity_vector(self, title, default=None):
        index = self._vocab.get_entity_index(title)
        if index:
            return self.entity_embedding[index]
        else:
            return default

    def get_text_vector(self, text):
        vectors = [self.get_word_vector(t.text.lower())
                   for t in self._tokenizer.tokenize(text)]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return None

        ret = np.mean(vectors, axis=0)
        ret = np.dot(ret, self._W)
        ret += self._b

        ret /= np.linalg.norm(ret, 2)

        return ret
