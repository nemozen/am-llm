'''AmBert class to facilitate using embedding generated in
bert_embedding. Requires metadata dictionary vocab_am.tsv and
embedding vectors embedding_am.tsv as output by bert_embedding.

'''
import csv
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_text as tf_text
from embedding.amparser import WORD_SEP


EMBEDDING_TSV=os.path.join(os.getenv('AM_LLM'), "embedding/embedding_am.tsv")
VOCAB_TSV=os.path.join(os.getenv('AM_LLM'), "embedding/vocab_am.tsv")
WEIGHTS_NPY=os.path.join(os.getenv('AM_LLM'), "embedding/embedding_am.npy")

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
# logger.setLevel(logging.INFO)

_ambert_instance = None


def _get_ambert_weights():
    """Load embedding (TSV of vectors) into a weight matrix.
    """
    try:
        logger.info("Loading weights from {} ...".format(WEIGHTS_NPY))
        weights = np.load(WEIGHTS_NPY)
        return weights
    except FileNotFoundError as e:
        logger.debug(e)
    logger.warn("Building embedding weights matrix, first time is slow...")
    logger.info("Loading weights from {} ...".format(EMBEDDING_TSV))
    linenum=0
    W = []
    with open(EMBEDDING_TSV) as csvfile:
        for row in csv.reader(csvfile, delimiter='\t'):
            v = list(map(float, row))
            W.append(v)
            linenum += 1
            if linenum % 10000 == 0:
                logger.debug("{} rows\r".format(linenum))

    weights = np.array(W)
    logger.debug("AmBert weights matrix: {}".format(weights.shape))
    np.save(WEIGHTS_NPY, weights)
    return weights


class AmBertMatrixInitializer(tf.keras.initializers.Initializer):

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, shape, dtype=None):
        assert shape == self.weights.shape
        assert dtype == 'float32'
        return self.weights

    def get_config(self):
        return {'weights': self.weights}


class AmBert():

    def __init__(self):
        self.weights = _get_ambert_weights()
        self.vocab_dict = {}
        with open(VOCAB_TSV) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                self.vocab_dict[row['word']] = int(row['id'])

    def convert_tokens_to_ids(self, tokens, min_length=None):
        ids = []
        for token in tokens.numpy()[0]:
            token = token.decode("utf-8")
            tid = self.vocab_dict.get(token,
                                      self.vocab_dict.get("[UNK]"))
            ids.append(tid)
        if min_length:
            ids += [self.vocab_dict.get("[PAD]")]*(min_length-len(ids))
        return ids

    def encode(self, sentence, min_length=None):
        """sentence -> list of word ids. Pad to min_length if specified."""
        tokenizer = tf_text.RegexSplitter(WORD_SEP)
        tokens = tokenizer.split(sentence)
        ids = self.convert_tokens_to_ids(tokens, min_length)
        return ids

    def get_embedding_layer(self, input_length=None):
        embedding_layer = tf.keras.layers.Embedding(
            self.weights.shape[0],  # vocab size
            self.weights.shape[1],  # embedding dims
            embeddings_initializer=AmBertMatrixInitializer(self.weights),
            input_length=input_length,
            trainable = False)
        return embedding_layer


def get_ambert_instance():
    global _ambert_instance
    if not _ambert_instance:
        _ambert_instance = AmBert()
    return _ambert_instance
