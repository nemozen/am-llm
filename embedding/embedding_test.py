#!/usr/bin/python3
from embedding.bert_embedding import Bert
from embedding.ambert_embedding import AmBert
import tensorflow as tf


bert = Bert()

def test_embedding_layer(s):
    """Output out of bert embedding layer should be the vectors in the
    bert weight matrix corresponding to the encoding, which should
    decoded in bert back to the input string tokens (without
    punctuation and all lower case).

    """
    print(s)
    v = bert.encode(s)
    print(v)
    res = bert.get_embedding_layer(len(v))(tf.convert_to_tensor([v]))
    assert res.shape == (1,len(v), bert.weights.shape[1])
    print(res[0])
    output = ' '.join(bert.decode(v.numpy()) for v in res[0])
    print(output)
    assert output == s
    print("PASSED")

test_embedding_layer("this is a sentence with seven words")


ambert = AmBert()

def test_am_embedding_layer(s, expected_output, length=None):
    """Output of ambert embedding layer should be the vectors representing
    the meaning. When decoded in bert it should give english tokens
    with meaning close to the input, word by word. But of course,
    sentence level translation won't make sense.

    """
    print(s)
    v = ambert.encode(s, length)
    print(v)
    res = ambert.get_embedding_layer(len(v))(tf.convert_to_tensor([v]))
    assert res.shape == (1,len(v), ambert.weights.shape[1])
    print(res[0])
    output = ' '.join(bert.decode(v.numpy()) for v in res[0])
    print(output)
    assert output == expected_output
    print("PASSED")

test_am_embedding_layer("ሰላም ዓለም", "hello world")
test_am_embedding_layer("የሴቶች እና የሕፃናት መብቶች", "women and children rights")
test_am_embedding_layer("ቢግ ማክ", "big mac [PAD] [PAD]", 4)
test_am_embedding_layer("የኢትዮጵያ ቋንቋ", "ethiopia language")
test_am_embedding_layer("ኮምጣጤ", "vinegar")
