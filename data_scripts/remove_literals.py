import os
import python_minifier
import re
import subprocess

rootDir = "/Users/paigefox/repos/artbaby/mindata2"


def walkFiles(root):
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            sourceFileName = dirName + "/" + fname
            print(sourceFileName)
            with open(sourceFileName, "w+") as f:
                try:
                    minified = python_minifier.minify(
                        f.read(), remove_literal_statements=True)
                    f.write(minified)
                except Exception as e:
                    print(e)
                f.close()

walkFiles(rootDir)
