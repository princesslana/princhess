import glob
import numpy
import random
import sklearn
import sys
from numpy import array2string
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier(
    loss="log",
    verbose=True,
    fit_intercept=False,
    n_jobs=3,
    eta0=0.01,
    learning_rate="constant",
    alpha=1e-6,
)


def train(classifier, files):
    all_files = list(glob.glob(files))

    random.shuffle(all_files)

    test_files = all_files[: int(len(all_files) / 10)]
    train_files = all_files[int(len(all_files) / 10) :]

    print("Loading test data...")
    x_test = None
    y_test = None
    for f in test_files:
        print(f"Loading {f}...")
        x, y = load_svmlight_file(f)

        x_test = vstack((x_test, x)) if x_test is not None else x
        y_test = numpy.concatenate((y_test, y)) if y_test is not None else y

    classes = numpy.unique(y_test)

    print(f"{len(y_test)} samples in test data")

    print(f"Training from {len(train_files)} files...")

    test_scores = []
    learning_rate = classifier.get_params()["eta0"]
    n_iters_no_improvement = 0
    best_coefs = None

    while n_iters_no_improvement < 3:
        print("Fitting...")
        random.shuffle(train_files)
        classifier.set_params(eta0=learning_rate)
        for i, f in enumerate(train_files):
            print(
                f"Loading {f} ({i + 1}/{len(train_files)}, iteration {len(test_scores) + 1})..."
            )
            x, y = load_svmlight_file(f)
            print(f"{len(y)} samples loaded")

            y_shuffled, x_shuffled = sklearn.utils.shuffle(y, x)

            classifier.partial_fit(x, y, classes=classes)

        print("Testing...")
        test_score = classifier.score(x_test, y_test)

        if test_scores:
            if test_score < max(test_scores):
                n_iters_no_improvement += 1
            else:
                n_iters_no_improvement = 0
                classifier.densify()
                best_coefs = classifier.coef_.transpose()
        else:
            classifier.densify()
            best_coefs = classifier.coef_.transpose()

        test_scores.append(test_score)
        learning_rate /= 2

        print(f"Test Scores: {test_scores}")

    return best_coefs


def write_coefs(file, coefs):
    numpy.set_printoptions(threshold=numpy.inf)

    with open(file, "w") as f:
        print(array2string(coefs, separator=","), file=f)


def train_state():
    coefs = train(classifier, files="model_data/*.libsvm.*")
    write_coefs("model", coefs)


def train_policy():
    coefs = train(classifier, files="policy_data/*.libsvm.*")
    coefs = numpy.ravel(coefs)
    write_coefs("policy_model", coefs)


train_what = sys.argv[1] if len(sys.argv) == 2 else None

if train_what == "state":
    train_state()
elif train_what == "policy":
    train_policy()
else:
    print("Must specify to train either 'state' or 'policy'")
