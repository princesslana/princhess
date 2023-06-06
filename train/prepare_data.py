import functools
import numpy
import sys

from sklearn.datasets import load_svmlight_file

MOVE_LABELS = 384


# Hacky way to make sure we flush the output buffer
print = functools.partial(print, flush=True)


def svm_to_npy(filename):
    print(f"Preparing {filename}...", end="")
    print("Loading...", end="")
    x, wdl = load_svmlight_file(filename)

    print("Splitting...", end="")
    moves, features = (
        x[:, :MOVE_LABELS],
        x[:, MOVE_LABELS:],
    )

    print("Saving...", end="")
    numpy.save(filename + ".npy", {"features": features, "wdl": wdl, "moves": moves})

    print("Done.")


for f in sys.argv[1:]:
    svm_to_npy(f)
