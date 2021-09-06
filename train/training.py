import glob
import numpy
import random
import sys
from numpy import array2string
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier

n_repeats = 2
n_iter = 50

classifier = SGDClassifier(loss='hinge', verbose=True, fit_intercept=False, n_jobs=-1)

files = glob.glob("model_data/*.libsvm.*")

print(f"Reading from {len(files)} files...")

files = list(files) * n_repeats
random.shuffle(files)

print("Fitting...")
for i, f in enumerate(files):
    print(f"Loading {f} ({i + 1}/{len(files)})...")
    x, y = load_svmlight_file(f)
    print(f"{len(y)} samples loaded", file=sys.stderr)

    for i in range(n_iter):
        print(f"Iteration {i + 1}/{n_iter}...")
        classifier.partial_fit(x, y, classes=numpy.unique(y))


classifier.densify()

numpy.set_printoptions(threshold=numpy.inf)

with open("model", "w") as f:
    print(array2string(classifier.coef_.transpose(), separator=","), file=f)






