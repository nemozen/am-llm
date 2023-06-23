#!/usr/bin/python3
'''
Utility to bootstrap different embeddings from BERT. The new vocab
is fed via stdin and we output vectors and/or a vocab dictionary with
encoding ids stdout.
'''
import argparse
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
from collections import OrderedDict
from official.nlp.bert import tokenization
from scipy.spatial.distance import cosine

# https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12
BERT_BASE=os.getenv('BERT_BASE')

# Special tokens and their ids in BERT embedding. In the output
# embeddings, we use the same tokens and give them sequential ids
# starting from 0, and the regular vocabulary token ids start after
# that. This dict is the form token -> (bert id, our id)
SPECIAL_TOKENS = OrderedDict([
    ("[PAD]", (0,0)),    # padding
    ("[UNK]", (100,1)),  # unknown word
    ("[CLS]", (101,2)),  # start of sentence
    ("[SEP]", (102,3))   # end of sentence
    ])

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
# Adjust logging level for this module
# logger.setLevel(logging.DEBUG)


def get_tokenizer():
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(BERT_BASE, "assets", "vocab.txt"),
        do_lower_case=True)
    return tokenizer


def get_bert_weights():
    """ Return weight matrix (list of vectors) of bert embedding"""
    bert_layer = hub.KerasLayer(BERT_BASE, trainable=True)
    W = bert_layer.get_weights()[0]
    assert len(get_tokenizer().vocab) == W.shape[0], "tokenizer and emedding size mismatch"
    logger.debug("bert weights matrix: {} {}".format(type(W), W.shape))
    return W


class BertMatrixInitializer(tf.keras.initializers.Initializer):

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, shape, dtype=None):
        assert shape == self.weights.shape
        assert dtype == 'float32'
        return self.weights

    def get_config(self):
        return {'weights': self.weights}


class Bert():

    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.weights = get_bert_weights()

    def encode(self, sentence):
        """sentence -> list of word ids"""
        tokens = self.tokenizer.tokenize(sentence)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def convert_vector_to_ids(self, v):
        """Return the ids of the top 5 vectors closest by cosine-similarity to v."""
        scores_ids = list(map(lambda x: (cosine(x[0], v), x[1]),
                              zip(self.weights,
                                  range(self.weights.shape[0]))))
        scores_ids.sort(key=lambda x: x[0], reverse=False)
        return [s[1] for s in scores_ids]

    def decode(self, v):
        """Decode one vector into one token, the most similar in BERT."""
        best_ids = self.convert_vector_to_ids(v)
        best_words = self.tokenizer.convert_ids_to_tokens(best_ids)
        logger.debug("decoded: {}".format(best_words[:5]))
        return best_words[0]

    def phrase_embedding_vector(self, phrase):
        """Given a string, returns a vector of floats of size embeddding_dims.

        Uses bert embedding on the word pieces to make a single
        vector.

        Uses simple vector addition: the vector for "foo bar" is the
        vector for "foo" + vector for "bar", which is ok in general
        https://medium.com/data-from-the-trenches/arithmetic-properties-of-word-embeddings-e918e3fda2ac
        but not always e.g. "children's" ends up closer to "s" than
        "children".
        # TODO: improve using output of full bert model instead of first layer weights
        """
        embedding_dims = self.weights.shape[1]
        vec = np.array([0]*embedding_dims)
        for i in self.encode(phrase):
            v = self.weights[i]
            vec = np.add(vec, v)

        return vec

    def get_embedding_layer(self, input_length=None):
        embedding_layer = tf.keras.layers.Embedding(
            self.weights.shape[0],  # vocab size
            self.weights.shape[1],  # embedding dims
            embeddings_initializer=BertMatrixInitializer(self.weights),
            input_length=input_length)
        return embedding_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dump TSVs of vocab or vector to be used in e.g. https://projector.tensorflow.org/")
    parser.add_argument("--vectors", action='store_true', help="TSV of vectors")
    parser.add_argument("--vocab", action='store_true', help="TSV of names and encoding ids (with header row)")
    args = parser.parse_args()

    # vector and vocab sequence must match so outputs work together in
    # e.g. ambert_embedding

    if args.vectors:
        bert = Bert()
        for wids in SPECIAL_TOKENS.values():
            print('\t'.join(str(x) for x in bert.weights[wids[0]]))
        for line in sys.stdin:
            row = bert.phrase_embedding_vector(line.strip())
            print('\t'.join([str(x) for x in row]))

    if args.vocab:
        print("word\tid")
        wid = -1
        for token,wids in SPECIAL_TOKENS.items():
            wid = wids[1]
            print('{}\t{}'.format(token, wid))
        for line in sys.stdin:
            wid += 1
            print('{}\t{}'.format(line.strip(), wid))
