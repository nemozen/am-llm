# Large Language Model for Amharic

Using text corpora from Amaric wikipedia and Ethiopian News Headlines,
we build AMBERT a language embedding bootstrapped off
[BERT](https://github.com/google-research/bert) and [Google
Translate](https://cloud.google.com/python/docs/reference/translate/latest/client). See
the [emedding](embedding/README.md) for details.

The resulting embedding can be used via the
[AmBert](embedding/ambert_embedding.py) class to build various NLP
applications.

[am2en](models/am2en.py) is a basic example of a model using this
embedding for an amharic to english translation application.

## Setup

`BERT_BASE` environment variable, which is used in
[bert_embedding](embedding/bert_embedding.py), should point to the BERT installation directory.

`AM_LLM` environment variable, which is used in [ambert_embedding](embedding/ambert_embedding.py),
should point to the am-llm repository's directory i.e. this directory.

Add `$AM_LLM` to your `PYTHONPATH` environment variable.

E.g.
```
export BERT_BASE=~/bert_base
export AM_LLM=~/src/am-llm
export PYTHONPATH=$AM_LLM:$PYTHONPATH
```