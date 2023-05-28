import numpy
import sys

from sklearn.datasets import load_svmlight_file


def svm_to_npy(filename):
    print(f"Preparing {filename}.npy...")
    x, y = load_svmlight_file(filename)
    numpy.save(filename + ".npy", {"features": x, "wdl": y})


for f in sys.argv[1:]:
    svm_to_npy(f)
