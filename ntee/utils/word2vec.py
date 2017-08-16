# -*- coding: utf-8 -*-

import bz2
import click
import joblib
import numpy as np
from contextlib import closing
from gensim.models.word2vec import Word2Vec, LineSentence
from marisa_trie import Trie
from multiprocessing.pool import Pool
from wiki_extractor import WikiExtractor
from wiki_dump_reader import WikiDumpReader

from tokenizer import RegexpTokenizer
from vocab import Vocab

MARKER = u'ENTITY/'

_extractor = None


def generate_corpus(dump_file, entity_db, out_file, abstract_db, learn_entity,
                    pool_size, chunk_size):
    dump_reader = WikiDumpReader(dump_file)

    global _extractor
    _extractor = WikiExtractor()

    with bz2.BZ2File(out_file, mode='w') as f:
        click.echo('Processing Wikipedia dump...')
        with closing(Pool(pool_size)) as pool:
            for paragraphs in pool.imap_unordered(
                _process_page, dump_reader, chunksize=chunk_size
            ):
                for paragraph in paragraphs:
                    para_words = paragraph.words

                    if learn_entity:
                        words = []
                        cur = 0
                        for link in sorted(paragraph.wiki_links, key=lambda l: l.span[0]):
                            title = entity_db.resolve_redirect(link.title).replace(u' ', u'_')
                            words += para_words[cur:link.span[0]]
                            words.append(MARKER + title)
                            cur = link.span[1]

                        words += para_words[cur:]

                    else:
                        words = para_words

                    f.write(u' '.join(words).encode('utf-8') + '\n')

        if abstract_db is not None:
            click.echo('Processing paragraphs in Abstract DB...')
            tokenizer = RegexpTokenizer()

            for value in abstract_db.itervalues():
                para_text = value['text']
                links = value['links']

                if learn_entity:
                    cur = len(para_text)
                    words = []
                    for (text, title, span) in sorted(links, key=lambda l: l[2][0], reverse=True):
                        words = ([MARKER + entity_db.resolve_redirect(title).replace(u' ', u'_')] +
                                 [t.text.lower() for t in tokenizer.tokenize(para_text[span[1]:cur])] +
                                 words)
                        cur = span[0]

                    words += [t.text.lower() for t in tokenizer.tokenize(para_text[:cur])] + words

                else:
                    words = [t.text.lower() for t in tokenizer.tokenize(para_text)]

                f.write(u' '.join(words).encode('utf-8') + '\n')


def train(corpus_file, out_file, mode, dim_size, window, min_count,
          negative, epoch, pool_size, chunk_size):
    with bz2.BZ2File(corpus_file) as f:
        sentences = LineSentence(f)
        sg = int(mode == 'sg')

        model = Word2Vec(sentences, size=dim_size, window=window, min_count=min_count,
                         workers=pool_size, iter=epoch, negative=negative, sg=sg)

    words = []
    entities = []
    for (w, _) in model.vocab.iteritems():
        if w.startswith(MARKER):
            entities.append(w[len(MARKER):].replace(u'_', u' '))
        else:
            words.append(w)

    vocab = Vocab(Trie(words), Trie(entities))

    word_embedding = np.zeros((len(words), dim_size), dtype=np.float32)
    entity_embedding = np.zeros((len(entities), dim_size), dtype=np.float32)
    for word in words:
        word_embedding[vocab.get_word_index(word)] = model[word]
    for entity in entities:
        entity_embedding[vocab.get_entity_index(entity)] = model[MARKER + entity.replace(u' ', u'_')]

    ret = dict(
        word_embedding=word_embedding,
        entity_embedding=entity_embedding,
        vocab=vocab,
    )
    joblib.dump(ret, out_file, compress=False)


def _process_page(page):
    return _extractor.extract_paragraphs(page)
