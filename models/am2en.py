#!/usr/bin/python3
import argparse
import csv
import logging
import numpy as np
import sys
import tensorflow as tf
from keras.callbacks import CSVLogger

from embedding.ambert_embedding import AmBert
from embedding.bert_embedding import Bert
from embedding.bert_embedding import SPECIAL_TOKENS


BATCH_SIZE=32
INPUT_WIDTH=10  # max length in tokens per row of input
OUTPUT_WIDTH=32  # max length in tokens per row of output
OUTPUT_TOKENS_TO_FILTER=["[PAD]"]

logger = logging.getLogger("am2en")
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
# Adjust logging level for this module
logger.setLevel(logging.DEBUG)


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
            xr = [SPECIAL_TOKENS["[CLS]"][1]] + ambert.encode(row) + \
                 [SPECIAL_TOKENS["[SEP]"][1]]
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
            yr = [SPECIAL_TOKENS["[CLS]"][0]] + bert.encode(row) + \
                 [SPECIAL_TOKENS["[SEP]"][0]]
            if len(yr) > OUTPUT_WIDTH:
                del x[xi]
                xi -= 1
                continue
            y.append(yr)

    # make rectangular
    for row in x:
        row += [SPECIAL_TOKENS["[PAD]"][1]]*(INPUT_WIDTH-len(row))

    for row in y:
        row += [SPECIAL_TOKENS.get("[PAD]")[0]]*(OUTPUT_WIDTH-len(row))

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


def build_attention_model(input_width, output_width):
    input_layer =tf.keras.Input(batch_size=BATCH_SIZE, shape=(input_width,))
    ambert_layer = ambert.get_embedding_layer(input_width)(input_layer)
    # TODO: add position encoding to embedding output
    attention_layers = []
    for i in range(output_width):
        if i == 0:
            query_ = ambert_layer
        else:
            query_ = attention_layers[i-1]
        query = tf.keras.layers.Dense(embedding_dims,
                                      name="query{}".format(i))(query_)
        value = tf.keras.layers.Dense(embedding_dims,
                                      name="value{}".format(i))(ambert_layer)
        key = tf.keras.layers.Dense(embedding_dims,
                                    name="key{}".format(i))(ambert_layer)
        attn = tf.keras.layers.Attention()([query, value, key])
        attention_layers.append(attn)

    attention = tf.keras.layers.Concatenate()(attention_layers)
    reduced = tf.keras.layers.GlobalAveragePooling1D()(attention)
    output_layer = tf.keras.layers.Reshape(
        (output_width, embedding_dims))(reduced)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def build_idp_model(input_width, output_width):
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
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Amharic to English translator.")
    parser.add_argument("--model", type=str, default="attention",
                        choices=["attention", "idp"], help="name of model to use")
    parser.add_argument("--train", type=int, default=0,
                        help="train model for this number of epochs")
    parser.add_argument("--predict", action='store_true', help="translate from stdin")
    args = parser.parse_args()

    build_model_function = "build_{}_model".format(args.model)
    model_name = "am2en_{}".format(args.model)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():

        model = globals()[build_model_function](INPUT_WIDTH, OUTPUT_WIDTH)
        tf.keras.utils.plot_model(model, to_file='{}.png'.format(model_name),
                                  show_shapes=True)

        try:
            model.load_weights('{}.ckpt'.format(model_name))
            logger.info("Loaded model weights from {}.ckpt".format(model_name))
        except tf.errors.NotFoundError as e:
            logger.info(e)

        if args.train:
            model.compile(loss="cosine_similarity")
            x, y = load_training_data(
                    "../../Amharic-English-Machine-Translation-Corpus/new-am.txt",
                    "../../Amharic-English-Machine-Translation-Corpus/new-en.txt")
            csv_logger = CSVLogger('log.csv', append=True, separator=';')
            model.fit(x, y, batch_size=BATCH_SIZE, epochs=args.train, verbose=True, callbacks=[csv_logger])
            model.save_weights('{}.ckpt'.format(model_name))
            logger.info("Saved model weights to {}.ckpt".format(model_name))

        if args.predict:
            for line in sys.stdin:
                v = [SPECIAL_TOKENS["[CLS]"][1]] + ambert.encode(line) + \
                    [SPECIAL_TOKENS["[SEP]"][1]]
                # break line up in chunks of size at most INPUT_WIDTH
                for i in range(1+int(len(v)/INPUT_WIDTH)):
                    x = v[i*INPUT_WIDTH:(i+1)*INPUT_WIDTH]
                    # pad to length INPUT_WIDTH
                    x += [SPECIAL_TOKENS.get("[PAD]")[1]]*(INPUT_WIDTH-len(x))
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
