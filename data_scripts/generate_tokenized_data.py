import os
from pprint import pprint
import re
import tokenize
from tokenize import STRING, NUMBER

root_dir = "/Users/paigefox/repos/artbaby/s3data"
tokenized_dir = "/Users/paigefox/repos/artbaby/s3data_tokens"

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
            if stop_at == 10:
                break
            stop_at += 1
            sourceFileName = dirName + "/" + fname
            print(sourceFileName)
            tokens = []
            with open(sourceFileName, "r") as f:
                tokens = tokenize.generate_tokens(f.readline)
                for token in tokens:
                    if token.type == STRING:
                        addToDict(string_literals, token.string)
                        addToDict(all_words, token.string.split(''))
                    elif token.type == NUMBER:
                        # Start of literal
                        tokens.append("SOL")
                        # Literal
                        tokens.append(token.string)
                        # End of literal
                        tokens.append("EOL")
                    else:
                        tokens.append(token.string)
                        addToDict(all_words, token.string)
            with open(tokenized_dir + "/" + fname, "w") as f:
                f.write(tokens)

    print("\n\n\n\nAll words:\n\n\n\n\n")
    pprint(all_words)
    print("\n\n\n\nString literals:\n\n\n\n\n")
    pprint(string_literals)
    return all_words


walkFiles(root_dir)
