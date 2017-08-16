Neural Text-Entity Encoder (NTEE)
=================================

## Introduction

Neural Text-Entity Encoder (NTEE) is a neural network model that learns embeddings (or distributed representations) of texts and Wikipedia entities.
Our model places a text and its relevant entities close to each other in a continuous vector space.
The details are explained in the paper [Learning Distributed Representations of Texts and Entities from Knowledge Base](https://arxiv.org/abs/1705.02494).

## Setup

The following commands install our code and its required libraries:

```
% pip install Cython
% pip install -r requirements.txt
% python setup.py develop
```

## Download Trained Embeddings

The embeddings used in our experiments can be downloaded from the following links:

* [ntee_300_sentence.joblib.gz](https://s3-ap-northeast-1.amazonaws.com/ntee/pub/models/ntee_300_sentence.joblib.gz) (300d vectors, 1.8GB, trained on Wikipedia sentences)
* [ntee_300_paragraph.joblib.gz](https://s3-ap-northeast-1.amazonaws.com/ntee/pub/models/ntee_300_paragraph.joblib.gz) (300d vectors, 1.8GB, trained on Wikipedia paragraphs)

These models are Python *dict* objects serialized with [joblib](https://pythonhosted.org/joblib/) and compressed with gzip.

If you want to use the embeddings in your program, please use `ntee.model_reader.ModelReader`:

```python
>>> from ntee.model_reader import ModelReader
>>> model = ModelReader('ntee_300_sentence.joblib')
>>> model.get_word_vector(u'apple')
memmap([ -1.81156114e-01,  -2.22634017e-01,  -8.77011120e-02,
        -1.41643256e-01,   2.06349805e-01,  -3.81092727e-01,
...
>>> model.get_entity_vector(u'Apple Inc.')
memmap([ -2.48675242e-01,  -1.21547781e-01,  -1.57411948e-01,
        -1.69242024e-01,   3.46656404e-02,  -2.03787461e-02,
...
>>> model.get_text_vector(u'Apple, orange, and banana')
array([ -1.90800596e-02,   8.16421525e-05,  -5.20865507e-02,
        -1.36841238e-02,   2.05799076e-03,   1.26077831e-02,
...
```

Also, you can directly de-serialize the model file using joblib:

```python
>>> import joblib
>>> model_obj = joblib.load('ntee_300_sentence.joblib')
>>> model_obj.keys()
['word_embedding', 'vocab', 'b', 'W', 'entity_embedding']
```

## Reproducing Sentence Similarity Experiments

**SICK:**

```bash
% wget "http://clic.cimec.unitn.it/composes/materials/SICK.zip"
% unzip SICK.zip
% ntee evaluate_sick ntee_300_sentence.joblib SICK.txt
0.7144 (pearson) 0.6046 (spearman)
```
**STS 2014:**

```bash
% wget "http://alt.qcri.org/semeval2014/task10/data/uploads/sts-en-gs-2014.zip"
% unzip sts-en-gs-2014.zip
% ntee evaluate_sts ntee_300_sentence.joblib sts-en-test-gs-2014
OnWN: 0.7204 (pearson) 0.7443 (spearman)
deft-forum: 0.5643 (pearson) 0.5490 (spearman)
deft-news: 0.7436 (pearson) 0.6775 (spearman)
headlines: 0.6876 (pearson) 0.6246 (spearman)
images: 0.8204 (pearson) 0.7671 (spearman)
tweet-news: 0.7467 (pearson) 0.6592 (spearman)
```

*NOTE*: The *ntee* command displays a *TypeError* warning due to the issue descibed [here](https://github.com/pallets/click/issues/564).

## Training Embeddings

This section describes how to train a new NTEE model from scratch.

**(1) Building Databases**

First, we need to download several files and build databases using these files.

```bash
% ntee download_dbpedia_abstract_files .
% wget https://s3-ap-northeast-1.amazonaws.com/ntee/pub/enwiki-20160601-pages-articles.xml.bz2
% ntee build_abstract_db . dbpedia_abstract.db
% ntee build_entity_db enwiki-20160601-pages-articles.xml.bz2 entity_db
% ntee build_vocab dbpedia_abstract.db entity_db vocab
```

**(2) Training Pre-trained Embeddings**

The pre-trained embeddings can be built using the following two commands:

```bash
% ntee word2vec generate_corpus enwiki-20160601-pages-articles.xml.bz2 entity_db word2vec_corpus.txt.bz2
% ntee word2vec train word2vec_corpus.txt.bz2 word2vec_sg_300.joblib
```

**(3) Training NTEE**

Now, we can start to train our NTEE embeddings.
The training takes approximately six days on NVIDIA K80 GPU.

```bash
% ntee train_model dbpedia_abstract.db entity_db vocab --word2vec=word2vec_sg_300.joblib ntee_paragraph.joblib
```

## Reference

If you use the code or the trained embedding in your research, please cite the following paper:

```
@article{yamada2017ntee,
  author    = {Yamada, Ikuya  and  Shindo, Hiroyuki  and  Takeda, Hideaki  and  Takefuji, Yoshiyasu},
  title     = {Learning Distributed Representations of Texts and Entities from Knowledge Base},
  journal   = {arXiv preprint arXiv:1705.02494},
  year      = {2017},
}
```


## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
