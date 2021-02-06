import os
from pprint import pprint
import python_minifier
import re
import subprocess

rootDir = "/Users/paigefox/repos/artbaby/mindata2"


def addToDict(words, key):
    if words.get(key) != None:
        words[key] += 1
    else:
        words[key] = 1


def walkFiles(root):
    all_imports = {}
    stop_at = 0
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            # if stop_at == 1000:
            #     break
            # stop_at += 1
            sourceFileName = dirName + "/" + fname
            # print(sourceFileName)
            with open(sourceFileName, "r") as f:
                word = ""
                for line in f.readlines():
                    if re.search("^import ", line) or re.search("^from .* import ", line):
                        addToDict(all_imports, line)
    return all_imports


pprint(walkFiles(rootDir))
