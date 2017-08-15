# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from contextlib import closing
from marisa_trie import Trie, RecordTrie
from multiprocessing.pool import Pool

from wiki_extractor import WikiExtractor
from wiki_dump_reader import WikiDumpReader

_extractor = None


class EntityDB(object):
    def __init__(self, title_dict, redirect_dict, inlink_arr):
        self._title_dict = title_dict
        self._redirect_dict = redirect_dict
        self._inlink_arr = inlink_arr

    def __len__(self):
        return len(self._title_dict)

    def __contains__(self, key):
        return key in self._title_dict

    def resolve_redirect(self, title):
        try:
            index = self._redirect_dict[title][0][0]
            title = self._title_dict.restore_key(index)

        except KeyError:
            pass

        return title

    def get_inlink_count(self, title, resolve_redirect=False):
        if resolve_redirect:
            title = self.resolve_redirect(title)

        try:
            index = self._title_dict[title]
            return self._inlink_arr[index]

        except KeyError:
            return 0

    @staticmethod
    def build(dump_file, pool_size, chunk_size):
        dump_reader = WikiDumpReader(dump_file)

        global _extractor
        _extractor = WikiExtractor()

        titles = []
        redirects = {}
        title_counter = Counter()

        with closing(Pool(pool_size)) as pool:
            for (page, links) in pool.imap_unordered(
                _process_page, dump_reader, chunksize=chunk_size
            ):
                titles.append(page.title)
                if page.is_redirect:
                    redirects[page.title] = page.redirect

                for link_obj in links:
                    title_counter[link_obj.title] += 1

        title_dict = Trie(titles)

        redirect_items = []
        for (title, dest_title) in redirects.iteritems():
            if dest_title in title_dict:
                redirect_items.append((title, (title_dict[dest_title],)))

        redirect_dict = RecordTrie('<I', redirect_items)

        for (title, count) in title_counter.items():
            dest_obj = redirect_dict.get(title)
            if dest_obj is not None:
                title_counter[title_dict.restore_key(dest_obj[0][0])] += count
                del title_counter[title]

        inlink_arr = np.zeros(len(title_dict), dtype=np.int)
        for (title, count) in title_counter.items():
            title_index = title_dict.get(title)
            if title_index is not None:
                inlink_arr[title_index] = count

        return EntityDB(title_dict, redirect_dict, inlink_arr)

    def save(self, out_file):
        self._title_dict.save(out_file + '_title.trie')
        self._redirect_dict.save(out_file + '_redirect.trie')
        np.save(out_file + '_prior.npy', self._inlink_arr)

    @staticmethod
    def load(in_file, mmap=True):
        title_dict = Trie()
        redirect_dict = RecordTrie('<I')

        title_dict.mmap(in_file + '_title.trie')
        redirect_dict.mmap(in_file + '_redirect.trie')
        inlink_arr = np.load(in_file + '_prior.npy', mmap_mode='r')

        return EntityDB(title_dict, redirect_dict, inlink_arr)


def _process_page(page):
    return (page, [l for p in _extractor.extract_paragraphs(page) for l in p.wiki_links])
