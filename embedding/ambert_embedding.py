import csv
import logging
import numpy as np
import sys
import tensorflow as tf
import tensorflow_text as tf_text

import amparser
import bert_embedding

TRANSLATED_VOCAB="vocab_en_test.txt"
VOCAB_TSV="vocab_am.tsv"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)


def get_ambert_weights():
    """Load translated vocabulary (words are translated into english
    phrases) and output phrase embedding using BERT (english).
    """
    logger.info("Computing weights for {} ...".format(TRANSLATED_VOCAB))
    bert = bert_embedding.Bert()
    linenum=0
    W = None
    for line in open(TRANSLATED_VOCAB):
        v = bert.phrase_embedding_vector(line)
        if W is None:
            W = np.array([v])
        else:
            W = np.append(W, [v], axis=0)

        linenum += 1
        if linenum % 1000 == 0:
            logger.info("{} rows".format(linenum))

    logger.info("AmBert weights matrix: {} {}".format(type(W), W.shape))
    return W


class AmBertMatrixInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        self.weights = get_ambert_weights()

    def __call__(self, shape, dtype=None):
        assert shape == self.weights.shape
        assert dtype == 'float32'
        return self.weights

    def get_config(self):
        return {'weights': self.weights}


class AmBert():

    def __init__(self):
        self.weights = get_ambert_weights()
        self.vocab_dict = {}
        with open(VOCAB_TSV) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                self.vocab_dict[row['word']] = int(row['id'])

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens.numpy()[0]:
            token = token.decode("utf-8")
            ids.append(self.vocab_dict.get(token, 0))
        return ids

    def encode(self, sentence):
        """sentence -> list of word ids"""
        tokenizer = tf_text.RegexSplitter(amparser.WORD_SEP)
        tokens = tokenizer.split(sentence)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def get_embedding_layer(self, input_length=None):
        embedding_layer = tf.keras.layers.Embedding(
            self.weights.shape[0],  # vocab size
            self.weights.shape[1],  # embedding dims
            embeddings_initializer=AmBertMatrixInitializer,
            input_length=input_length)
        return embedding_layer
