import bert_embedding
import ambert_embedding
import tensorflow as tf


bert = bert_embedding.Bert()

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


ambert = ambert_embedding.AmBert()

def test_am_embedding_layer(s, expected_output):
    """Output of ambert embedding layer should be the vectors representing
    the meaning. When decoded in bert it should give english tokens
    with meaning close to the input words.

    """
    print(s)
    v = ambert.encode(s)
    print(v)
    res = ambert.get_embedding_layer(len(v))(tf.convert_to_tensor([v]))
    assert res.shape == (1,len(v), ambert.weights.shape[1])
    print(res[0])
    output = ' '.join(bert.decode(v.numpy()) for v in res[0])
    print(output)
    assert output == expected_output
    print("PASSED")

test_am_embedding_layer("ሰላም ዓለም", "hello world")
# TODO: poor normalization in phrase embedding: "children's" is closer
# to "s" than "children",
test_am_embedding_layer("የሕፃናትና የሴቶች መብቶች ጉዳይ የሁሉም ኃላፊነት ነው",
                        "s women rights matter all responsibility it")
