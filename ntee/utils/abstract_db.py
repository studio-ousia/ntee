# -*- coding: utf-8 -*-

import click
import gzip
import os
import rdflib
import re
import urllib
from collections import defaultdict
from contextlib import closing
from functools import partial
from multiprocessing.pool import Pool
from shelve import DbfilenameShelf

from tokenizer import RegexpTokenizer


class AbstractDB(DbfilenameShelf):
    def __init__(self, *args, **kwargs):
        DbfilenameShelf.__init__(self, *args, **kwargs)

    @staticmethod
    def build(in_dir, out_file, pool_size):
        with closing(AbstractDB(out_file, protocol=-1)) as db:
            target_files = [f for f in sorted(os.listdir(in_dir)) if f.endswith('ttl.gz')]
            with closing(Pool(pool_size)) as pool:
                f = partial(_process_file, in_dir=in_dir)
                for ret in pool.imap(f, target_files):
                    for (key, obj) in ret:
                        db[key] = obj

    def count_valid_words(self, vocab, max_text_len):
        tokenizer = RegexpTokenizer()
        keys = self.keys()
        words = frozenset(list(vocab.words()))
        word_count = 0

        with click.progressbar(keys) as bar:
            for key in bar:
                c = 0
                for token in tokenizer.tokenize(self[key]['text']):
                    if token.text.lower() in words:
                        c += 1

                word_count += min(c, max_text_len)

        return word_count


def _process_file(file_name, in_dir):
    abs_matcher = re.compile(ur'^http://dbpedia\.org/resource/(.*)/abstract#offset_(\d+)_(\d+)$')
    dbp_matcher = re.compile(ur'^http://dbpedia\.org/resource/(.*)$')

    click.echo('Processing %s' % file_name)

    g = rdflib.Graph()
    with gzip.GzipFile(os.path.join(in_dir, file_name)) as f:
        g.load(f, format='turtle')

    texts = {}
    mentions = defaultdict(dict)
    mention_titles = defaultdict(dict)
    for (s, p, o) in g:
        s = unicode(s)
        p = unicode(p)
        o = unicode(o)

        abs_match_obj = abs_matcher.match(s)
        title = urllib.unquote(urllib.unquote(abs_match_obj.group(1).encode('utf-8')))
        title = title.decode('utf-8').replace(u'_', u' ')

        if p == u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString':
            texts[title] = o

        elif p == u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf':
            span = (int(abs_match_obj.group(2)), int(abs_match_obj.group(3)))
            mentions[title][s] = (o, span)

        elif p == u'http://www.w3.org/2005/11/its/rdf#taIdentRef':
            match_obj = dbp_matcher.match(o)
            if match_obj:
                link_title = urllib.unquote(match_obj.group(1).encode('utf-8'))
                link_title = link_title.decode('utf-8').replace(u'_', u' ')
                mention_titles[title][s] = link_title

    ret = []
    for (title, text) in texts.iteritems():
        links = []
        for (key, link_title) in mention_titles[title].items():
            (name, span) = mentions[title][key]
            links.append((name, link_title, span))

        ret.append((title.encode('utf-8'),
                    dict(title=title, text=text, links=links)))

    return ret
