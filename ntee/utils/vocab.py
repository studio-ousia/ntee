# -*- coding: utf-8 -*-

import click
from collections import Counter
from marisa_trie import Trie

from tokenizer import RegexpTokenizer


class Vocab(object):
    __slots__ = ('_word_dict', '_entity_dict')

    def __init__(self, word_dict, entity_dict):
        self._word_dict = word_dict
        self._entity_dict = entity_dict

    @property
    def word_size(self):
        return len(self._word_dict)

    @property
    def entity_size(self):
        return len(self._entity_dict)

    def words(self):
        return iter(self._word_dict)

    def entities(self):
        return iter(self._entity_dict)

    def get_word_index(self, word, default=None):
        try:
            return self._word_dict.key_id(word)
        except KeyError:
            return default

    def get_entity_index(self, entity, default=None):
        try:
            return self._entity_dict.key_id(entity)
        except KeyError:
            return default

    def get_word_by_index(self, index):
        return self._word_dict.restore_key(index)

    def get_entity_by_index(self, index):
        return self._entity_dict.restore_key(index)

    @staticmethod
    def build(db, entity_db, min_word_count, min_entity_count):
        word_counter = Counter()
        entity_counter = Counter()

        tokenizer = RegexpTokenizer()
        with click.progressbar(db.keys()) as bar:
            for title in bar:
                obj = db[title]
                text = obj['text']
                tokens = tokenizer.tokenize(text)

                word_counter.update(t.text.lower() for t in tokens)

                for (_, title, _) in obj['links']:
                    title = entity_db.resolve_redirect(title)
                    entity_counter[title] += 1

        word_dict = Trie([w for (w, c) in word_counter.iteritems()
                          if c >= min_word_count])
        entity_dict = Trie([e for (e, c) in entity_counter.iteritems()
                            if c >= min_entity_count])

        return Vocab(word_dict, entity_dict)

    def save(self, out_file):
        self._word_dict.save(out_file + '_word.trie')
        self._entity_dict.save(out_file + '_entity.trie')

    @staticmethod
    def load(in_file, mmap=True):
        word_dict = Trie()
        entity_dict = Trie()

        word_dict.mmap(in_file + '_word.trie')
        entity_dict.mmap(in_file + '_entity.trie')

        return Vocab(word_dict, entity_dict)

    def __reduce__(self):
        return (self.__class__, (self._word_dict, self._entity_dict))
