import csv
import logging
import numpy as np
import sys
import tensorflow as tf

from embedding.ambert_embedding import AmBert
from embedding.bert_embedding import Bert


BATCH_SIZE=1
MAX_INPUT_WIDTH=10  # max length in words per row of input

logger = logging.getLogger("am2en")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)

ambert = AmBert()
bert = Bert()
embedding_dims = bert.weights.shape[1]


def load_training_data(xfile, yfile):
    rows_to_keep = []
    with open(xfile) as infile:
        x = []
        xlen = 0
        i = 0
        for row in infile:
            i += 1
            xr = ambert.encode(row)
            if len(xr) > MAX_INPUT_WIDTH:
                continue
            rows_to_keep.append(i)
            x.append(xr)
            xlen = max(len(xr), xlen)

    with open(yfile) as infile:
        y = []
        ylen = 0
        i = 0
        for row in infile:
            i += 1
            if i not in rows_to_keep:
                continue
            yr = bert.encode(row)
            y.append(yr)
            ylen = max(len(yr), ylen)

    # make rectangular
    for row in x:
        row += [0]*(xlen-len(row))

    for row in y:
        row += [0]*(ylen-len(row))

    # batch of size 1
    for i in range(len(x)):
        x[i] = [x[i]]

    for i in range(len(y)):
        y[i] = [bert.get_embedding_layer()(tf.convert_to_tensor(y[i]))]

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    logger.debug("Input shape: {}\nOutput shape: {}".format(x.shape, y.shape))
    return x,y


def build_model(input_width, output_width):
    input_layer =tf.keras.Input(batch_size=BATCH_SIZE, shape=(1,None))
    ambert_layer = ambert.get_embedding_layer(input_width)(input_layer)
    flatten_layer = tf.keras.layers.Reshape(
        (1, embedding_dims*input_width))(ambert_layer)
    dense_layer = tf.keras.layers.Dense(
        embedding_dims*output_width, activation='relu')(flatten_layer)
    output_layer = tf.keras.layers.Reshape(
        (output_width, embedding_dims))(dense_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model


x, y = load_training_data(
    "../../Amharic-English-Machine-Translation-Corpus/am_test.txt",
    "../../Amharic-English-Machine-Translation-Corpus/en_test.txt")
model = build_model(x.shape[2], y.shape[2])

try:
    model.load_weights('{}.ckpt'.format("am2en"))
    logger.info("Loaded model from {}.ckpt".format("am2en"))
except tf.errors.NotFoundError as e:
    model.compile(loss="cosine_similarity")
    model.fit(x, y, batch_size=1, epochs=2, verbose=True)
    model.save_weights('{}.ckpt'.format("am2en"))

for line in sys.stdin:
    v = [[ambert.encode(line)]]
    logger.debug("Input shape: {}".format(v))
    res = model.predict(v)[0]
    logger.debug(res.shape)
    output = ' '.join(bert.decode(w) for w in res)
    print(output)

