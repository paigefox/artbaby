import os
from pprint import pprint
import re
import subprocess

rootDir = "/Users/paigefox/repos/artbaby/mindata2"

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
    stop_at = 0
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            if stop_at == 10:
                break
            stop_at += 1
            sourceFileName = dirName + "/" + fname
            print(sourceFileName)
            with open(sourceFileName, "r") as f:
                word = ""
                for c in f.read():
                    if (c in code_delimiters or c in math_delimiters or c in string_delimiters):
                        if word != "":
                            addToDict(all_words, word)
                        addToDict(all_words, c)
                        word = ""
                    else:
                        word += c
    return all_words


pprint(walkFiles(rootDir))
