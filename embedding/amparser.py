#!/usr/bin/python3

import os
import re
import sys
from html.parser import HTMLParser

# whitespace, ascii punctuation, ge'ez punctuation, other punctuation, arabic numberals
WORD_SEP=r"\s|[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]|[።፡፣፤፥፦፧፨፠]|[‘‚“”•…«»]|[0-9]"


def get_html_files(dirname):
    for dirpath,dirs,files in os.walk(dirname):
        for fname in files:
            fpath = os.path.join(dirpath,fname)
            if not os.path.isfile(fpath):
                continue
            if fname.split('.')[-1] != 'html':
                continue
            yield fpath


class WordsFromHTML(HTMLParser):
    words = set()

    def handle_data(self, data):
        self.words.update(re.split(WORD_SEP, data))


def get_file_text(fpath):
    parser = WordsFromHTML()
    for line in open(fpath):
        parser.feed(line)
    return parser.words
    
    
if __name__ == "__main__":
    words = set()
    num_files = 0
    for fpath in get_html_files(sys.argv[1]):
        num_files += 1
        words.update(get_file_text(fpath))

    sys.stderr.write("%d files, %d terms\n"%(num_files, len(words)))
    for w in words:
        print(w)
