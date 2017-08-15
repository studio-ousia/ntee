# -*- coding: utf-8 -*-

import click
import os
import re
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr

from ntee.cli import cli
from ntee.model_reader import ModelReader


@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('dataset_file', type=click.File())
def evaluate_sick(model_file, dataset_file):
    reader = ModelReader(model_file)

    predicted = []
    correct = []
    for (n, line) in enumerate(dataset_file):
        if n == 0:
            continue

        data = line.rstrip().decode('utf-8').split('\t')
        sent1 = data[1]
        sent2 = data[2]
        score = float(data[4])
        fold = data[11]
        if fold == 'TRIAL':
            continue

        correct.append(float(score))

        vec1 = reader.get_text_vector(sent1)
        vec2 = reader.get_text_vector(sent2)
        predicted.append(1.0 - cosine(vec1, vec2))

    click.echo('%.4f (pearson) %.4f (spearman)' % (
        pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0]
    ))


@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('dataset_dir', type=click.Path(exists=True))
def evaluate_sts(model_file, dataset_dir):
    reader = ModelReader(model_file)

    for file_name in sorted(os.listdir(dataset_dir)):
        match_obj = re.match(r'^STS\.input\.(.*)\.txt', file_name)
        if not match_obj:
            continue

        name = match_obj.group(1)

        predicted = []
        correct = []
        with open(os.path.join(dataset_dir, file_name)) as input_file:
            with open(os.path.join(dataset_dir, 'STS.gs.' + name + '.txt')) as gs_file:
                for (line, score) in zip(input_file, gs_file):
                    score = score.rstrip()
                    if not score:
                        continue
                    score = float(score)

                    (sent1, sent2) = line.rstrip().decode('utf-8').split('\t')
                    correct.append(score)

                    vec1 = reader.get_text_vector(sent1)
                    vec2 = reader.get_text_vector(sent2)

                    predicted.append(1.0 - cosine(vec1, vec2))

        click.echo('%s: %.4f (pearson) %.4f (spearman)' % (
            name, pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0]
        ))
