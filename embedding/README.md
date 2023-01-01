# Download text corpora

Download text corpora into  `../../am_corpora`. We use:

. Amharic wikipedia corpus, [amwiki](https://archive.org/download/amwiki-20190101/amwiki-20190101-pages-meta-current.xml.bz2)

. Ethiopian News Headlines corpus, [enh-corpus](https://github.com/geezorg/enh-corpus/releases/)

Another source, which is not used in our examples currently, is [CACO](https://www.findke.ovgu.de/findke/en/Research/Data+Sets/Contemporary+Amharic+Corpus+(CACO)-p-1142.html) which can be downloaded from [here](http://wwwiti.cs.uni-magdeburg.de/iti_dke/Datasets/Contemporary_Amharic_Corpus_(CACO)-version_1.1.zip).

# Extract vocabulary from the XML and HTML files in the corpora
```
./amparser.py "../../am_corpora" | sort - > vocab_am.txt
```
This output is included in this repo, [vocab_am.txt](vocab_am.txt), made by running amparser.py on enh-corpus and amwiki combined.

# Translate words into english phrases
```
cat vocab_am.txt | ./translate.py > vocab_en.txt
```
This output is included in this repo, [vocab_en.txt](vocab_en.txt).

# Create metadata dict: table of vocab words with encoding ids
```
cat vocab_am.txt | ./bert_embedding.py --vocab > vocab_am.tsv
```
# Use BERT to compute AMBERT embedding vectors
```
cat vocab_en.txt | ./bert_embedding.py --vectors > embedding_am.tsv
```
# Load AmBert layer and test
```
./embedding_test.py
```