# Extract vocabulary from corpus
```
./amparser.py ".." | sort - > vocab_am.txt
```
This output is included in this repo, [vocab_am.txt](vocab_am.txt), made by running amparser.py on enh-corpus and amwiki combined.

# Translate words into english phrases
```
cat vocab_am.txt | ./translate.py > vocab_en.txt
```
This output is included in this repo, [vocab_en.txt](vocab_en.txt).

# create metadata dict: table of vocab words with encoding ids
```
cat vocab_am.txt | ./bert_embedding.py --vocab > vocab_am.tsv
```
# Use BERT to create AM-BERT embedding
```
cat vocab_en.txt | ./bert_embedding.py --vectors > embedding_am.tsv
```
# Test
```
./embedding_test.py
```