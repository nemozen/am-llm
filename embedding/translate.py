#!/usr/bin/python3

import six
import sys
from google.cloud import translate_v2 as translate


def translate_text(text, src_lang, dest_lang):
    """Translates text from src_lang to dest_lang.

    src_lang and dest_lang must be ISO 639-1 language codes.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text, target_language=dest_lang, source_language=src_lang)
    return result["translatedText"]


if __name__ == "__main__":

    for line in sys.stdin:
        print(translate_text(line.strip(), "am", "en"))
