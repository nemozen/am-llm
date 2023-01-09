#!/usr/bin/python3
import argparse
import csv
import logging
import numpy as np
import sys
import tensorflow as tf

from embedding.ambert_embedding import AmBert
from embedding.bert_embedding import Bert


BATCH_SIZE=64
INPUT_WIDTH=10  # max length in tokens per row of input
OUTPUT_WIDTH=32  # max length in tokens per row of output
MODEL_NAME="am2en_idp"
OUTPUT_TOKENS_TO_FILTER=["[PAD]"]

logger = logging.getLogger("am2en")
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
# Adjust logging level for this module
# logger.setLevel(logging.INFO)


ambert = AmBert()
bert = Bert()
embedding_dims = bert.weights.shape[1]


def load_training_data(xfile, yfile):
    logger.info("Loading training data...")
    rows_to_keep = []
    with open(xfile) as infile:
        x = []
        i = -1
        for row in infile:
            i += 1
            xr = ambert.encode(row)
            if len(xr) > INPUT_WIDTH:
                continue
            rows_to_keep.append(i)
            x.append(xr)

    with open(yfile) as infile:
        y = []
        i = -1
        xi = -1
        for row in infile:
            i += 1
            if i not in rows_to_keep:
                continue
            xi += 1
            yr = bert.encode(row)
            if len(yr) > OUTPUT_WIDTH:
                del x[xi]
                xi -= 1
                continue
            y.append(yr)

    # make rectangular
    for row in x:
        row += [0]*(INPUT_WIDTH-len(row))

    for row in y:
        row += [0]*(OUTPUT_WIDTH-len(row))

    for i in range(len(y)):
        y[i] = bert.get_embedding_layer()(tf.convert_to_tensor(y[i]))

    # truncate to multiple of batch size
    x = x[:(len(x) - len(x) % BATCH_SIZE)]
    y = y[:(len(y) - len(y) % BATCH_SIZE)]

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    logger.info("Input shape: {}\nOutput shape: {}".format(
        x.shape, y.shape))
    return x,y


def build_model(input_width, output_width):
    '''Four layers: ambert embedding, flattening, dense layer, and output
    layer.  The AMBERT embedding is designed to be semantically
    related to the BERT english embedding used to decode the output,
    i.e. vectors in the input embedding will be close to the vectors
    with similar meaning in the output embedding. So we initialize the
    dense layer in the middle with the identity kernel.
    '''
    input_layer =tf.keras.Input(batch_size=BATCH_SIZE, shape=(input_width,))
    ambert_layer = ambert.get_embedding_layer(input_width)(input_layer)
    flatten_layer = tf.keras.layers.Reshape(
        (1, embedding_dims*input_width))(ambert_layer)
    dense_layer = tf.keras.layers.Dense(
        embedding_dims*output_width,
        activation='relu',
        kernel_initializer='identity')(flatten_layer)
    output_layer = tf.keras.layers.Reshape(
        (output_width, embedding_dims))(dense_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    # tf.keras.utils.plot_model(model, to_file='{}.png'.format(MODEL_NAME), show_shapes=True)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Amharic to English translator.")
    parser.add_argument("--train", type=int, default=0,
                        help="train model for this number of epochs")
    parser.add_argument("--predict", action='store_true', help="translate from stdin")
    args = parser.parse_args()

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():

        model = build_model(INPUT_WIDTH, OUTPUT_WIDTH)

        try:
            model.load_weights('{}.ckpt'.format(MODEL_NAME))
            logger.info("Loaded model weights from {}.ckpt".format(MODEL_NAME))
        except tf.errors.NotFoundError as e:
            logger.info(e)

        if args.train:
            model.compile(loss="cosine_similarity")
            x, y = load_training_data(
                    "../../Amharic-English-Machine-Translation-Corpus/new-am.txt",
                    "../../Amharic-English-Machine-Translation-Corpus/new-en.txt")
            model.fit(x, y, batch_size=BATCH_SIZE, epochs=args.train, verbose=True)
            model.save_weights('{}.ckpt'.format(MODEL_NAME))
            logger.info("Saved model weights to {}.ckpt".format(MODEL_NAME))

        if args.predict:
            for line in sys.stdin:
                v = ambert.encode(line)
                # break line up in chunks of size at most INPUT_WIDTH
                for i in range(1+int(len(v)/INPUT_WIDTH)):
                    x = v[i*INPUT_WIDTH:(i+1)*INPUT_WIDTH]
                    # pad to length INPUT_WIDTH
                    x += [0]*(INPUT_WIDTH-len(x))
                    x = tf.convert_to_tensor([x], dtype=tf.float32)
                    logger.debug("Encoded input: {} {}".format(x.shape, x))
                    res = model(x)
                    logger.debug("Result shape: {}".format(res.shape))
                    for r in res:
                        output = ' '.join(
                            filter(lambda token: token not in OUTPUT_TOKENS_TO_FILTER,
                                   (bert.decode(w) for w in r)))
                        print(output, end=' ')  # Line fragment
                print('')  # EOL
