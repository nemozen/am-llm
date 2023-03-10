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
OUTPUT_WIDTH=20  # max length in tokens per row of output
OUTPUT_TOKENS_TO_FILTER=["[PAD]"]
ATTN_HEADS=8  # number of "heads" in multihead attention model


logger = logging.getLogger("am2en")
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
# Set logging level for this module
logger.setLevel(logging.INFO)

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


def build_scaled_transformer_model(input_width, output_width):
    '''Attention model based on
    [Vaswami et al, 2017]
    (https://arxiv.org/abs/1706.03762)
    [The Illustrated transformer]
    (https://jalammar.github.io/illustrated-transformer/)
    [Attention is all you need: Discovering the Transformer paper]
    (https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)
    '''
    # dimensionality of query, key, value
    attn_dims = embedding_dims/ATTN_HEADS

    input_layer =tf.keras.Input(batch_size=BATCH_SIZE,
                                shape=(input_width,))
    embedding = ambert.get_embedding_layer(input_width)(input_layer)
    # TODO: add position encoding to embedding output

    # ENCODER

    # attention
    attention_heads = []
    for i in range(ATTN_HEADS):
        query = tf.keras.layers.Dense(attn_dims,
                                      name="query{}".format(i))(embedding)
        value = tf.keras.layers.Dense(attn_dims,
                                      name="value{}".format(i))(embedding)
        key = tf.keras.layers.Dense(attn_dims,
                                    name="key{}".format(i))(embedding)
        attn = tf.keras.layers.Attention(use_scale=True)([query, value, key])
        attention_heads.append(attn)

    mhattention = tf.keras.layers.Concatenate()(attention_heads)
    # trainable linear combination of heads
    attention = tf.keras.layers.Dense(embedding_dims,
                                      name="enc_attention_linear")(mhattention)
    # residual sum and layer normalization along embedding dimension, i.e. last axis
    xpz = tf.keras.layers.Add()([embedding, attention])
    norm_attention = tf.keras.layers.LayerNormalization()(xpz)

    # feed forward
    enc_ff = tf.keras.layers.Dense(units=embedding_dims, name="enc_ff",
                                   activation="relu")(norm_attention)
    # TODO: add linear layer to FF

    # normalize after feed forward
    xpz2 = tf.keras.layers.Add()([norm_attention, enc_ff])
    encoder_out = tf.keras.layers.LayerNormalization()(xpz2)

    # DECODER

    # self attention
    prev_layer_out = encoder_out  # if stacked, this is previous layer's output
    decoder_attention_heads = []
    for i in range(ATTN_HEADS):
        query = tf.keras.layers.Dense(attn_dims,
                                      name="ds_query{}".format(i))(prev_layer_out)
        value = tf.keras.layers.Dense(attn_dims,
                                      name="ds_value{}".format(i))(prev_layer_out)
        key = tf.keras.layers.Dense(attn_dims,
                                    name="ds_key{}".format(i))(prev_layer_out)
        attn = tf.keras.layers.Attention(use_scale=True)(
            [query, value, key], use_causal_mask=True)
        decoder_attention_heads.append(attn)

    dec_mhattn = tf.keras.layers.Concatenate()(decoder_attention_heads)
    # trainable linear combination of heads
    decoder_attention = tf.keras.layers.Dense(
        embedding_dims, name="dec_attention_linear")(dec_mhattn)
    # residual sum and layer normalization along embedding dimension, i.e. last axis
    dec_xpz = tf.keras.layers.Add()([encoder_out, decoder_attention])
    norm_dec_attention = tf.keras.layers.LayerNormalization()(dec_xpz)

    # encoder-decoder attention
    query_in = norm_dec_attention  # if stacked, this is previous layer output
    encdec_attention_heads = []
    for i in range(ATTN_HEADS):
        query = tf.keras.layers.Dense(attn_dims,
                                      name="ec_query{}".format(i))(query_in)
        value = tf.keras.layers.Dense(attn_dims,
                                      name="ec_value{}".format(i))(encoder_out)
        key = tf.keras.layers.Dense(attn_dims,
                                    name="ec_key{}".format(i))(encoder_out)
        attn = tf.keras.layers.Attention(use_scale=True)([query, value, key])  # mask/shift right?
        encdec_attention_heads.append(attn)

    encdec_mhattn = tf.keras.layers.Concatenate()(encdec_attention_heads)
    # trainable linear combination of heads
    encdec_attention = tf.keras.layers.Dense(
        embedding_dims, name="encdec_attention_linear")(encdec_mhattn)
    # residual sum and layer normalization along embedding dimension, i.e. last axis
    encdec_xpz = tf.keras.layers.Add()([norm_dec_attention, encdec_attention])
    norm_encdec_attention = tf.keras.layers.LayerNormalization()(encdec_xpz)

    # feed forward
    decoder_ff = tf.keras.layers.Dense(units=embedding_dims, name="dec_ff",
                                       activation="relu")(norm_encdec_attention)
    # normalize after feed forward
    dec_xpz2 = tf.keras.layers.Add()([norm_encdec_attention, decoder_ff])
    norm_decoder_ff = tf.keras.layers.LayerNormalization()(dec_xpz2)

    # LINEAR

    dec_flat = tf.keras.layers.Reshape((1, input_width*embedding_dims))(norm_decoder_ff)
    out_linear = tf.keras.layers.Dense(embedding_dims*output_width)(dec_flat)
    output_layer = tf.keras.layers.Reshape((output_width, embedding_dims))(out_linear)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def build_attention_model(input_width, output_width):
    '''Attention model, with a single multi-head dot-product attention
    layer.

    '''
    ATTN_DIM=64
    input_layer =tf.keras.Input(batch_size=BATCH_SIZE,
                                shape=(input_width,))
    embedding = ambert.get_embedding_layer(input_width)(input_layer)
    # TODO: add position encoding to embedding output

    # ENCODER

    # attention
    attention_heads = []
    for i in range(ATTN_HEADS):
        if i == 0:
            query_ = embedding
        else:
            query_ = attention_heads[i-1]
        query = tf.keras.layers.Dense(ATTN_DIM,
                                      name="query{}".format(i))(query_)
        value = tf.keras.layers.Dense(ATTN_DIM,
                                      name="value{}".format(i))(embedding)
        key = tf.keras.layers.Dense(ATTN_DIM,
                                    name="key{}".format(i))(embedding)
        attn = tf.keras.layers.Attention()([query, value, key])
        attention_heads.append(attn)

    mhattention = tf.keras.layers.Concatenate()(attention_heads)
    # trainable linear combination of heads
    attention = tf.keras.layers.Dense(embedding_dims, name="enc_attention")(mhattention)
    # residual sum and layer normalization along embedding dimension, i.e. last axis
    xpz = tf.keras.layers.Add()([embedding, attention])
    norm_attention = tf.keras.layers.LayerNormalization()(xpz)

    # feed forward
    ff1 = tf.keras.layers.Dense(units=embedding_dims, name="ff1",
                                activation="relu")(norm_attention)
    # normalize after feed forward
    xpz2 = tf.keras.layers.Add()([norm_attention, ff1])
    norm_ff = tf.keras.layers.LayerNormalization()(xpz2)

    # DECODER

    decoder_in = tf.keras.layers.Reshape((1, input_width*embedding_dims))(norm_ff)
    out1 = tf.keras.layers.Dense(embedding_dims*output_width)(decoder_in)
    norm_out = tf.keras.layers.LayerNormalization()(out1)

    output_layer = tf.keras.layers.Reshape((output_width, embedding_dims))(norm_out)
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
    embedding = ambert.get_embedding_layer(input_width)(input_layer)
    flatten_layer = tf.keras.layers.Reshape(
        (1, embedding_dims*input_width))(embedding)
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
    parser.add_argument("--model", type=str, default="scaled_transformer",
                        choices=["scaled_transformer", "attention", "idp"],
                        help="name of model to use")
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
            model.load_weights('{}.ckpt'.format(model_name)).expect_partial()
            logger.info("Loaded model weights from {}.ckpt".format(model_name))
        except tf.errors.NotFoundError as e:
            logger.info(e)

        if args.train:
            model.compile(loss="cosine_similarity")
            x, y = load_training_data(
                    "../../Amharic-English-Machine-Translation-Corpus/new-am.txt",
                    "../../Amharic-English-Machine-Translation-Corpus/new-en.txt")
            csv_logger = CSVLogger('{}_log.csv'.format(model_name), append=True, separator='\t')
            model.fit(x, y, batch_size=BATCH_SIZE, epochs=args.train, verbose=True, callbacks=[csv_logger])
            model.save_weights('{}.ckpt'.format(model_name))
            logger.info("Saved model weights to {}.ckpt".format(model_name))

        if args.predict:
            for line in sys.stdin:
                if not line.strip():
                    print(line, end='')  # echo empty line
                    continue
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
                    # TODO use softmax layer in model instead of bert.decode
                    for r in res:
                        output = ' '.join(
                            filter(lambda token: token not in OUTPUT_TOKENS_TO_FILTER,
                                   (bert.decode(w) for w in r)))
                        print(output, end=' ')  # Line fragment
                print('')  # EOL
