# Large Language Model for Amharic

Using text corpora from e.g. Amaric wikipedia [amwiki](amwiki), and
Ethiopian News Headlines [enh-corpus](enh-corpus), we build AMBERT, a language
embedding bootstrapped off
[BERT](https://github.com/google-research/bert) and [Google Translate](https://cloud.google.com/python/docs/reference/translate/latest/client).

The resulting embedding can be used via the [AmBert](embedding/ambert_embedding.py) class

[am2en](models/am2en.py) is a basic example of a model using this
embedding for an amharic to english translation application.

To execute the scripts, add this directory (/path/to/am-llm) to your PYTHONPATH
environment variable.
