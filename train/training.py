import glob
import numpy
import random
import sklearn
import sys

from concurrent.futures import ThreadPoolExecutor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import array2string
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import softmax_cross_entropy_with_logits

INPUT_SIZE = 768 + (5 * 64)

DEFAULT_BATCH_SIZE = 16384
DEFAULT_HIDDEN_LAYERS = 128


def generate_batches(files, batch_size=DEFAULT_BATCH_SIZE, categories=None):
    all_files = files[:]

    prev_svmlight_file = None
    next_svmlight_file = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            random.shuffle(all_files)
            for fname in all_files:
                prev_svmlight_file = next_svmlight_file
                next_svmlight_file = executor.submit(lambda: load_svmlight_file(fname))

                if prev_svmlight_file:
                    x, y = prev_svmlight_file.result()

                    output, input = sklearn.utils.shuffle(y, x)

                    # tweak batch size so we don't have a smaller batch left over
                    batches = input.shape[0] // batch_size
                    actual_batch_size = input.shape[0] // batches + 1

                    for local_index in range(0, input.shape[0], actual_batch_size):
                        input_local = input[local_index : (local_index + batch_size)]
                        output_local = output[local_index : (local_index + batch_size)]

                        if categories:
                            output_local = keras.utils.to_categorical(
                                output_local, num_classes=categories
                            )

                        yield input_local.todense(), output_local


def build_model(
    hidden_layers=DEFAULT_HIDDEN_LAYERS, *, output_layers, output_activation
):
    model = keras.Sequential()
    model.add(keras.Input(shape=(INPUT_SIZE,)))
    model.add(
        layers.Dense(hidden_layers, activation="relu", kernel_initializer="he_normal")
    )
    model.add(
        layers.Dense(
            output_layers,
            activation=output_activation,
            kernel_initializer="he_normal",
            use_bias=False,
        )
    )
    model.summary()

    return model


def train_state(files, model, start_epoch):
    train_files = list(glob.glob(files))
    train_generator = generate_batches(files=train_files)

    if not model:
        model = build_model(output_layers=1, output_activation="tanh")
    else:
        model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="mean_squared_error")

    mc = ModelCheckpoint(
        filepath="checkpoints/state."
        + str(INPUT_SIZE)
        + "x"
        + str(DEFAULT_HIDDEN_LAYERS)
        + "x1.e{epoch:03d}-l{loss:.2f}.h5",
        verbose=True,
    )

    model.fit(
        train_generator,
        epochs=100,
        verbose=1,
        callbacks=[mc],
        steps_per_epoch=1000,
        initial_epoch=start_epoch,
    )


def train_policy(files, model, start_epoch):
    train_files = list(glob.glob(files))
    train_generator = generate_batches(files=train_files, categories=1792)

    if not model:
        model = build_model(output_layers=1792, output_activation="linear")
    else:
        model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss=softmax_cross_entropy_with_logits, metrics=["acc"]
    )

    mc = ModelCheckpoint(
        filepath="checkpoints/policy.768x"
        + str(DEFAULT_HIDDEN_LAYERS)
        + "x1792.e{epoch:03d}-l{loss:.2f}-a{acc:.2f}.h5",
        verbose=True,
    )

    model.fit(
        train_generator,
        epochs=100,
        verbose=1,
        callbacks=[mc],
        steps_per_epoch=1000,
        initial_epoch=start_epoch,
    )


train_what = sys.argv[1] if len(sys.argv) >= 2 else None
model_file = sys.argv[2] if len(sys.argv) >= 3 else None
start_epoch = int(sys.argv[3]) - 1 if len(sys.argv) >= 4 else 0

model = None

if model_file:
    model = keras.models.load_model(
        model_file,
        custom_objects=dict(
            softmax_cross_entropy_with_logits_v2=softmax_cross_entropy_with_logits
        ),
    )


if train_what == "state":
    train_state("model_data/*.libsvm.*", model, start_epoch)
elif train_what == "policy":
    train_policy("policy_data/*.libsvm.*", model, start_epoch)
else:
    print("Must specify to train either 'state' or 'policy'")
