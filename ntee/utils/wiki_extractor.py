# -*- coding: utf-8 -*-

import logging
import mwparserfromhell

from tokenizer import RegexpTokenizer

logger = logging.getLogger(__name__)


class Paragraph(object):
    __slots__ = ('text', 'words', 'wiki_links')

    def __init__(self, text, words, wiki_links):
        self.text = text
        self.words = words
        self.wiki_links = wiki_links

    def __repr__(self):
        return '<Paragraph %s>' % (u' '.join(self.words[:5]).encode('utf-8') + '...')

    def __reduce__(self):
        return (self.__class__, (self.text, self.words, self.wiki_links))


class WikiLink(object):
    __slots__ = ('title', 'text', 'span')

    def __init__(self, title, text, span):
        self.title = title
        self.text = text
        self.span = span

    def __repr__(self):
        return '<WikiLink %s->%s>' % (self.text.encode('utf-8'),
                                      self.title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.span))


class WikiExtractor(object):
    def __init__(self, lowercase=True, min_paragraph_len=20):
        self._lowercase = lowercase
        self._min_paragraph_len = min_paragraph_len
        self._tokenizer = RegexpTokenizer()

    def extract_paragraphs(self, page):
        paragraphs = []
        cur_text = []
        cur_words = []
        cur_links = []

        if page.is_redirect:
            return []

        for node in self._parse_page(page).nodes:
            if isinstance(node, mwparserfromhell.nodes.Text):
                for (n, paragraph) in enumerate(unicode(node).split('\n')):
                    words = self._extract_words(paragraph)

                    if n == 0:
                        cur_text.append(paragraph)
                        cur_words += words
                    else:
                        paragraphs.append(
                            Paragraph(u' '.join(cur_text), cur_words, cur_links)
                        )
                        cur_text = [paragraph]
                        cur_words = words
                        cur_links = []

            elif isinstance(node, mwparserfromhell.nodes.Wikilink):
                title = node.title.strip_code()
                if not title:
                    continue

                if node.text:
                    text = node.text.strip_code()
                else:
                    text = node.title.strip_code()

                cur_text.append(text)
                words = self._extract_words(text)
                start = len(cur_words)
                cur_words += words
                end = len(cur_words)
                cur_links.append(
                    WikiLink(self._normalize_title(title), text, (start, end))
                )

            elif isinstance(node, mwparserfromhell.nodes.Tag):
                if node.tag not in ('b', 'i'):
                    continue
                if not node.contents:
                    continue

                text = node.contents.strip_code()
                cur_text.append(text)
                cur_words += self._extract_words(text)

        return [p for p in paragraphs
                if (p.words and (p.words[0] not in ('|', '!', '{')) and
                    len(p.words) >= self._min_paragraph_len)]

    def _extract_words(self, text):
        tokens = self._tokenizer.tokenize(text)

        if self._lowercase:
            words = [token.text.lower() for token in tokens]
        else:
            words = [token.text for token in tokens]

        return words

    def _parse_page(self, page):
        try:
            return mwparserfromhell.parse(page.wiki_text)
        except Exception:
            logger.exception('Failed to parse wiki text: %s', page.title)
            return mwparserfromhell.parse('')

    def _normalize_title(self, title):
        return (title[0].upper() + title[1:]).replace('_', ' ')
