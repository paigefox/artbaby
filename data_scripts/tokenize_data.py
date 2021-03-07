import csv
import os
from pprint import pprint
import re
import tokenize
from tokenize import NUMBER, STRING

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
    addToDict(all_words, "SOL")
    addToDict(all_words, "EOL")
    stop_at = 0
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            # if stop_at == 10:
            #     break
            # stop_at += 1
            sourceFileName = dirName + "/" + fname
            # print(sourceFileName)
            tokenized_data = []
            with open(sourceFileName, "r") as f:
                tokens = tokenize.generate_tokens(f.readline)
                for token in tokens:
                    if token.type == STRING:
                        for char in token.string:
                            tokenized_data.append(char)
                            addToDict(all_words, char)
                    elif token.type == NUMBER:
                        # Designate the number as a literal
                        tokenized_data.append("SOL")
                        tokenized_data.append(token.string)
                        tokenized_data.append("EOL")
                    else:
                        tokenized_data.append(token.string)
                        addToDict(all_words, token.string)
            tokenized_file = open(
                "tokenized/" + fname, "w")
            wr = csv.writer(tokenized_file, quoting=csv.QUOTE_ALL)
            wr.writerow(tokenized_data)
            tokenized_file.close()
    pprint(all_words)

walkFiles(rootDir)
