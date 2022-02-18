import glob
import numpy
import random
import sklearn
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import array2string
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

classifier = SGDClassifier(
    loss="log",
    verbose=True,
    fit_intercept=False,
    n_jobs=3,
    eta0=0.01,
    learning_rate="constant",
    alpha=1e-6,
)


def split_files_train_and_test(files):
    all_files = list(files)

    random.shuffle(all_files)

    split_idx = min(max(1, len(all_files) // 10), 5)
    test_files = all_files[:split_idx]
    train_files = all_files[split_idx:]

    return train_files, test_files


def generate_batches(files, batch_size):
    all_files = files[:]

    while True:
        random.shuffle(all_files)
        for fname in all_files:
            x, y = load_svmlight_file(fname)

            output, input = sklearn.utils.shuffle(y, x)

            for local_index in range(0, input.shape[0], batch_size):
                input_local = input[local_index : (local_index + batch_size)]
                output_local = output[local_index : (local_index + batch_size)]

                yield input_local.todense(), output_local


def load_files(fs):
    all_x = None
    all_y = None
    for f in fs:
        print(f"Loading {f}...")
        file_x, file_y = load_svmlight_file(f)

        all_x = vstack((all_x, file_x)) if all_x is not None else file_x
        all_y = numpy.concatenate((all_y, file_y)) if all_y is not None else file_y

    print(f"{len(all_y)} samples loaded.")
    return (all_x.todense(), all_y)


def train_state_with_keras(files):
    train_files = list(glob.glob(files))
    batch_size = 256
    train_generator = generate_batches(files=train_files, batch_size=batch_size)

    hidden_layers = 32

    model = keras.Sequential()
    model.add(keras.Input(shape=(768,)))
    model.add(
        layers.Dense(hidden_layers, activation="relu", kernel_initializer="he_normal")
    )
    model.add(
        layers.Dense(
            1, activation="tanh", kernel_initializer="he_normal", use_bias=False
        )
    )
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="mean_squared_error")

    mc = ModelCheckpoint(
        filepath="checkpoints/state.768x"
        + str(hidden_layers)
        + "x1.e{epoch:03d}-l{loss:.2f}.h5",
        verbose=True,
    )

    model.fit(
        train_generator,
        epochs=100,
        verbose=1,
        callbacks=[mc],
        steps_per_epoch=len(train_files) * 1000000 / batch_size,
    )


def train_policy_with_keras(key, files):
    train_files, test_files = split_files_train_and_test(list(glob.glob(files)))
    batch_size = 256
    test_data = load_files(test_files)
    train_generator = generate_batches(files=train_files, batch_size=batch_size)

    hidden_layers = 512 
    output_activation = "softmax"

    model = keras.Sequential()
    model.add(keras.Input(shape=(768,)))
    model.add(
        layers.Dense(hidden_layers, activation="relu", kernel_initializer="he_normal")
    )
    model.add(
        layers.Dense(
            4096, activation=output_activation, kernel_initializer="he_normal", use_bias=False
        )
    )
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"]
    )

    mc = ModelCheckpoint(
        filepath="checkpoints/"
        + key
        + ".768x"
        + str(hidden_layers)
        + "x4096"
        + output_activation
        + ".e{epoch:03d}-l{val_loss:.2f}-a{val_acc:.2f}.h5",
        verbose=True,
        monitor="val_loss",
        save_best_only=True,
    )
    es = EarlyStopping(monitor="val_loss", patience=15, verbose=True)

    model.fit(
        train_generator,
        epochs=500,
        verbose=1,
        callbacks=[mc, es],
        steps_per_epoch=len(train_files) * 1000000 / batch_size,
        validation_data=test_data,
    )


def train_state():
    train_state_with_keras(files="model_data/*.libsvm.*")


def train_policy(folder):
    train_policy_with_keras(folder, files=f"{folder}/*.libsvm.*")


train_what = sys.argv[1] if len(sys.argv) == 2 else None

if train_what == "state":
    train_state()
elif train_what == "from":
    train_policy("from_data")
elif train_what == "to":
    train_policy("to_data")
elif train_what == "policy":
    train_policy("policy_data")
else:
    print("Must specify to train either 'state' or 'policy'")
