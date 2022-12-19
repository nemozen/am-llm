'''Use pre-trained BERT to make an embedding for input text.

'''
import argparse
import logging
import numpy as np
import os
import scipy.linalg
import sys
import tensorflow_hub as hub
from official.nlp.bert import tokenization


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)

# https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12
BERT_BASE_DIR=os.path.join(os.getenv('HOME'), "bert_base")


def get_tokenizer():
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(BERT_BASE_DIR, "assets", "vocab.txt"),
        do_lower_case=True)
    return tokenizer


def get_bert_weights():
    """ Return  weight matrix (list of vectors) of bert embedding"""
    bert_layer = hub.KerasLayer(BERT_BASE_DIR, trainable=True)
    W = bert_layer.get_weights()[0]
    assert len(get_tokenizer().vocab) == W.shape[0], "tokenizer and emedding size mismatch"
    logger.info("bert weights matrix: {} {}".format(type(W), W.shape))
    return W


class Bert():

    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.bw = get_bert_weights()

    def encode(self, sentence):
        """sentence -> list of word ids"""
        tokens = self.tokenizer.tokenize(sentence)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids


    def phrase_embedding_vector(self, phrase):
        """Given a phrase, returns a vector of floats of size
        embeddding_dims. 

        Uses bert embedding on the word pieces to make a single
        vector, using simple vector addition, e.g. the vector for "foo
        bar" is the vector for "foo" + vector for "bar".
        https://medium.com/data-from-the-trenches/arithmetic-properties-of-word-embeddings-e918e3fda2ac

        Vector is Normalized to length 1.0 so that trained NNs don't
        get biased toward long names.

        """

        embedding_dims = self.bw.shape[1]
        vec = np.array([0]*embedding_dims)
        for i in self.encode(phrase):
            v = self.bw[i]
            vec = np.add(vec, v)

        vec_norm = scipy.linalg.norm(vec)
        if vec_norm > 0:
            return np.divide(vec, vec_norm)
        else:
            return vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dump TSVs of vocab or vector to be used in e.g. https://projector.tensorflow.org/")
    parser.add_argument("--vectors", action='store_true', help="TSV of vectors")
    parser.add_argument("--vocab", action='store_true', help="TSV of names and encoding ids (with header row)")
    args = parser.parse_args()

    if args.vectors:
        bert = Bert()
        for line in sys.stdin:
            row = bert.phrase_embedding_vector(line.strip())
            print('\t'.join([str(x) for x in row]))

    if args.vocab:
        print("word\tid")
        wid = 0
        for line in sys.stdin:
            wid += 1
            print('{}\t{}'.format(line.strip(), wid))
