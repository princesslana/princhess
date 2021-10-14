import glob
import numpy
import random
import sklearn
import sys
from numpy import array2string
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV

n_repeats = 3
n_iter = 1
classifier = SGDClassifier(
    loss="log",
    verbose=True,
    fit_intercept=False,
    n_jobs=-1,
    eta0=0.01,
    learning_rate="constant",
    alpha=1e-6,
)

files = glob.glob("policy_data/*.libsvm.*")
files = list(files)
random.shuffle(files)

test_files = files[: int(len(files) / 10)]
train_files = files[int(len(files) / 10) :]

print(f"Training from {len(train_files)} files...")

test_scores = []
learning_rate = 0.01

for r in range(n_repeats):
    print("Fitting...")
    random.shuffle(train_files)
    classifier.set_params(eta0=learning_rate)
    for i, f in enumerate(train_files):
        print(f"Loading {f} ({i + 1}/{len(train_files)}, {r + 1}/{n_repeats})...")
        x, y = load_svmlight_file(f)
        print(f"{len(y)} samples loaded", file=sys.stderr)

        for i in range(n_iter):
            print(f"Iteration {i + 1}/{n_iter}...")
            y_shuffled, x_shuffled = sklearn.utils.shuffle(y, x)
            classifier.partial_fit(x_shuffled, y_shuffled, classes=numpy.unique(y))

    print("Testing...")
    x_test = None
    y_test = None
    for f in test_files:
        print(f"Loading {f}...")
        x, y = load_svmlight_file(f)

        x_test = vstack((x_test, x)) if x_test is not None else x
        y_test = numpy.concatenate((y_test, y)) if y_test is not None else y

    print(f"{len(y_test)} samples in test data")

    test_score = classifier.score(x_test, y_test)

    print(f"Test Score: {test_score}")
    test_scores.append(test_score)

    learning_rate /= 2

print(f"Test Scores: {test_scores}")

classifier.densify()

numpy.set_printoptions(threshold=numpy.inf)

with open("policy_model", "w") as f:
    print(array2string(classifier.coef_.transpose(), separator=","), file=f)
