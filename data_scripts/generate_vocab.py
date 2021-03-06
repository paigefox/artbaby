import os
from pprint import pprint
import re
import tokenize
from tokenize import STRING

rootDir = "/Users/paigefox/repos/artbaby/s3data"

code_delimiters = set(
    [' ', '\t', '\n', '(', ')', '[', ']', ',', ':', '.', '=', ';'])
string_delimiters = set(['\'', '"', '`'])
math_delimiters = set(['+', '-', '/', '%', '*'])


def addToDict(words, key):
    if words.get(key) != None:
        words[key] += 1
    else:
        words[key] = 1


def walkFiles(root):
    all_words = {}
    string_literals = {}
    stop_at = 0
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            # if stop_at == 10:
            #     break
            # stop_at += 1
            sourceFileName = dirName + "/" + fname
            print(sourceFileName)
            with open(sourceFileName, "r") as f:
                tokens = tokenize.generate_tokens(f.readline)
                for token in tokens:
                    if token.type == STRING:
                        addToDict(string_literals, token.string)
                    else:
                        addToDict(all_words, token.string)
    print("\n\n\n\nAll words:\n\n\n\n\n")
    pprint(all_words)
    print("\n\n\n\nString literals:\n\n\n\n\n")
    pprint(string_literals)
    return all_words


walkFiles(rootDir)
