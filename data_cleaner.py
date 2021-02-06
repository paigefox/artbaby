import os
import python_minifier
import re
import subprocess

rootDir = "/Users/paigefox/Downloads/py150_files/data"


def walkDirectories(root):
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            if re.search("\.py$", fname):
                sourceFileName = dirName + "/" + fname
                with open(sourceFileName) as f:
                    try:
                        print(sourceFileName)
                        minified = python_minifier.minify(
                            f.read(), remove_literal_statements=True)
                        minfile = open(
                            "mindata2/" + re.sub("/", "_", sourceFileName), "w")
                        minfile.write(minified)
                        minfile.close()
                    except Exception as ex:
                        print("error processing file " +
                              dirName + "/" + fname + " converting to python 3 ", ex)
                        bashCmd = ["2to3", "-w", sourceFileName]
                        process = subprocess.Popen(
                            bashCmd, stdout=subprocess.PIPE)
                        output, error = process.communicate()
                        if output != "":
                            try:
                                minified = python_minifier.minify(
                                    f.read(), remove_literal_statements=True)
                                minfile = open(
                                    "mindata2/" + re.sub("/", "_", sourceFileName), "w")
                                minfile.write(minified)
                                minfile.close()
                            except Exception as e:
                                print("True exception " +
                                      sourceFileName + " " + e)
                    f.close()

walkDirectories(rootDir)
