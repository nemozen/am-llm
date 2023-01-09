# Preliminary steps

## Download BERT

```
mkdir $BERT_BASE
cd $BERT_BASE
wget https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4?tf-hub-format=compressed -O bert_en_uncased_L-12_H-768_A-12.tgz
tar -xzvf bert_en_uncased_L-12_H-768_A-12.tgz
```

## Download text corpora

Download text corpora into  `../../am_corpora`. We use:

- Amharic wikipedia corpus, [amwiki](https://dumps.wikimedia.org/amwiki). We used
[amwiki-20190101](https://archive.org/download/amwiki-20190101/amwiki-20190101-pages-meta-current.xml.bz2) but if starting from scratch, you might want the [latest](https://dumps.wikimedia.org/amwiki/latest/amwiki-latest-pages-meta-current.xml.bz2)

- Ethiopian News Headlines corpus, [enh-corpus](https://github.com/geezorg/enh-corpus/releases/)

Another source, which is not used in our examples currently, is [CACO](https://www.findke.ovgu.de/findke/en/Research/Data+Sets/Contemporary+Amharic+Corpus+(CACO)-p-1142.html) which can be downloaded from [here](http://wwwiti.cs.uni-magdeburg.de/iti_dke/Datasets/Contemporary_Amharic_Corpus_(CACO)-version_1.1.zip).

## Build vocabulary

Extract vocabulary from the XML and HTML files in the corpora, then translate words into english phrases.

```
./amparser.py "../../am_corpora" | sort - > vocab_am.txt
cat vocab_am.txt | ./translate.py > vocab_en.txt
```

The outputs are included in this repo: [vocab_am.txt](vocab_am.txt), made by running amparser.py on enh-corpus and amwiki combined; and [vocab_en.txt](vocab_en.txt), made by running the vocabulary through Google translate.

# Generate embedding files

## Create metadata dict: table of vocab words with encoding ids
```
cat vocab_am.txt | ./bert_embedding.py --vocab > vocab_am.tsv
```
## Use BERT to compute AMBERT embedding vectors
```
cat vocab_en.txt | ./bert_embedding.py --vectors > embedding_am.tsv
```

## Test Bert and AmBert classes
```
./embedding_test.py
```

## Visualize

Metadata and vector files (vocab_am.tsv and embedding_am.tsv as generated above) can be loaded in [Embedding Projector](https://projector.tensorflow.org/)

![](embedding_projector_viz.png)