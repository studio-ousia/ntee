# -*- coding: utf-8 -*-

import re


class Token:
    __slots__ = ('text', 'span')

    def __init__(self, text, span):
        self.text = text
        self.span = span

    def __repr__(self):
        return '<Token %s>' % self.text.encode('utf-8')

    def __reduce__(self):
        return (self.__class__, (self.text, self.span))


class RegexpTokenizer(object):
    __slots__ = ('_rule',)

    def __init__(self, rule=ur'[\w\d]+'):
        self._rule = re.compile(rule, re.UNICODE)

    def tokenize(self, text):
        return [Token(text[o.start():o.end()], o.span()) for o in self._rule.finditer(text)]
