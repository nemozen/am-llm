import logging
import sys
from tensorflow import convert_to_tensor, float32
from embedding.ambert_embedding import get_ambert_instance
from embedding.bert_embedding import get_bert_instance
from embedding.bert_embedding import SPECIAL_TOKENS


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ambert = get_ambert_instance()
bert = get_bert_instance()


def load_training_data(xfile, yfile, input_width, output_width, batch_size):
    logger.info("Loading training data...")
    # TODO: use tf dataset
    rows_to_keep = []
    with open(xfile) as infile:
        x = []
        i = -1
        for row in infile:
            i += 1
            xr = [SPECIAL_TOKENS["[CLS]"][1]] + ambert.encode(row) + \
                 [SPECIAL_TOKENS["[SEP]"][1]]
            if len(xr) > input_width:
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
            if len(yr) > output_width:
                del x[xi]
                xi -= 1
                continue
            y.append(yr)

    # make rectangular
    for row in x:
        row += [SPECIAL_TOKENS["[PAD]"][1]]*(input_width-len(row))

    for row in y:
        row += [SPECIAL_TOKENS.get("[PAD]")[0]]*(output_width-len(row))

    for i in range(len(y)):
        y[i] = bert.get_embedding_layer()(convert_to_tensor(y[i]))

    # truncate to multiple of batch size
    x = x[:(len(x) - len(x) % batch_size)]
    y = y[:(len(y) - len(y) % batch_size)]

    x = convert_to_tensor(x, dtype=float32)
    y = convert_to_tensor(y, dtype=float32)
    logger.info("Input shape: {}\nOutput shape: {}".format(
        x.shape, y.shape))
    return x,y
