from embedding.ambert_embedding import AmBert
from embedding.bert_embedding import Bert
import tensorflow as tf
import csv
import numpy as np

INPUT_WIDTH=169
OUTPUT_WIDTH=263
BATCH_SIZE=1

ambert = AmBert()
bert = Bert()
embedding_dims = bert.weights.shape[1]

input_layer =tf.keras.Input(batch_size=BATCH_SIZE, shape=(1,None))
ambert_layer = ambert.get_embedding_layer(INPUT_WIDTH)(input_layer)
flatten_layer = tf.keras.layers.Reshape((BATCH_SIZE,
                                         embedding_dims*INPUT_WIDTH))(ambert_layer)
dense_layer = tf.keras.layers.Dense(embedding_dims*OUTPUT_WIDTH,
                                    activation='relu')(flatten_layer)
output_layer = tf.keras.layers.Reshape((OUTPUT_WIDTH,
                                        embedding_dims))(dense_layer)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


with open("/tmp/parallel_text.txt") as infile:
    x = []
    y = []
    xlen = 0
    ylen = 0
    for row in csv.reader(infile, delimiter='\t'):
        xr = ambert.encode(row[0])
        x.append(xr)
        xlen = max(len(xr), xlen)
        yr = bert.encode(row[1])
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
print(x.shape, y.shape)
model.compile(loss="cosine_similarity")
model.fit(x, y, batch_size=1, epochs=100, verbose=True)

v = ambert.encode("የኢትዮጵያ መከላከያ ሠራዊት ከሐሙስ አንስቶ መቀሌ በመዝለቅ መደበኛ ኃላፊነቱን መወጣት ይጀምራል")
print(v)
res = model.predict(v)
print(res.shape)
output = ' '.join(bert.decode(w) for w in res)
print(output)


