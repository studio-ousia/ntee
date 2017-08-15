# -*- coding: utf-8 -*-

import click
import commands
import logging
import multiprocessing


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


import sentence_similarity
import train
from model_reader import ModelReader
from utils.abstract_db import AbstractDB
from utils import word2vec
from utils.entity_db import EntityDB
from utils.vocab import Vocab


@cli.command()
@click.argument('out_dir', type=click.Path(exists=True, file_okay=False))
def download_dbpedia_abstract_files(out_dir):
    for n in range(114):
        url = 'https://s3-ap-northeast-1.amazonaws.com/ntee/pub/dbpedia_abstract/abstracts_en%d.ttl.gz' % (n,)
        click.echo('Getting %s' % url)
        commands.getoutput('wget -P %s/ %s' % (out_dir, url))


@cli.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=10)
def build_abstract_db(**kwargs):
    AbstractDB.build(**kwargs)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def build_entity_db(dump_file, out_file, **kwargs):
    db = EntityDB.build(dump_file, **kwargs)
    db.save(out_file)


@cli.command()
@click.argument('db_file', type=click.Path())
@click.argument('entity_db_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--min-word-count', default=5)
@click.option('--min-entity-count', default=3)
def build_vocab(db_file, entity_db_file, out_file, **kwargs):
    db = AbstractDB(db_file, 'r')
    entity_db = EntityDB.load(entity_db_file)
    vocab = Vocab.build(db, entity_db, **kwargs)
    vocab.save(out_file)


@cli.group(name='word2vec')
def word2vec_group():
    pass


@word2vec_group.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('entity_db_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--learn-entity/--no-entity', default=True)
@click.option('--abstract-db', type=click.Path(), default=None)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def generate_corpus(dump_file, entity_db_file, out_file, abstract_db, **kwargs):
    entity_db = EntityDB.load(entity_db_file)
    if abstract_db:
        abstract_db = AbstractDB(abstract_db, 'r')

    word2vec.generate_corpus(dump_file, entity_db, out_file, abstract_db, **kwargs)


@word2vec_group.command(name='train')
@click.argument('corpus_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--mode', type=click.Choice(['sg', 'cbow']), default='sg')
@click.option('--dim-size', default=300)
@click.option('--window', default=10)
@click.option('--min-count', default=3)
@click.option('--negative', default=5)
@click.option('--epoch', default=5)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def train_word2vec(corpus_file, out_file, **kwargs):
    word2vec.train(corpus_file, out_file, **kwargs)


@cli.command()
@click.argument('db_file', type=click.Path())
@click.argument('entity_db_file', type=click.Path())
@click.argument('vocab_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--word2vec', type=click.Path())
@click.option('--mode', type=click.Choice(['paragraph', 'sentence']), default='paragraph')
@click.option('--text-len', default=2000)
@click.option('--dim-size', default=300)
@click.option('--negative', default=30)
@click.option('--epoch', default=1)
@click.option('--batch-size', default=100)
@click.option('--word-static', is_flag=True)
@click.option('--entity-static', is_flag=True)
@click.option('--include-title/--no-title', default=True)
@click.option('--optimizer', default='rmsprop')
@click.option('--dev-size', default=1000)
@click.option('--patience', default=1)
@click.option('--num-links', type=int)
@click.option('--random-seed', default=0)
def train_model(db_file, entity_db_file, vocab_file, word2vec, **kwargs):
    db = AbstractDB(db_file, 'r')
    entity_db = EntityDB.load(entity_db_file)
    vocab = Vocab.load(vocab_file)

    if word2vec:
        w2vec = ModelReader(word2vec)
    else:
        w2vec = None

    train.train(db, entity_db, vocab, w2vec, **kwargs)
