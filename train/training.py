import glob
import numpy
import random
import sklearn
import sys
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import array2string
from scipy.sparse import vstack
from tensorflow import keras
from tensorflow.keras import activations, layers, regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.nn import softmax_cross_entropy_with_logits

# pieces + last capture + threats
INPUT_SIZE = 768 + (5 * 64) + 768
HIDDEN_LAYERS = 192

ENTRIES_PER_FILE = 1000000

TRAIN_TIME_MINUTES = 8 * 60

STATE_EOPCH_TIME_MINUTES = 12
POLICY_EPOCH_TIME_MINUTES = 18

STATE_EPOCHS = TRAIN_TIME_MINUTES // STATE_EOPCH_TIME_MINUTES
POLICY_EPOCHS = TRAIN_TIME_MINUTES // POLICY_EPOCH_TIME_MINUTES

BATCH_SIZE = 16384


def load_npy_file(fname, values):
    data = numpy.load(fname, allow_pickle=True)

    x = data.item().get("features")
    y = values(data.item())

    out, inp = sklearn.utils.shuffle(y, x)

    return out, inp


def generate_npy_batches(files, values):
    all_files = files[:]

    prev_file = None
    next_file = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            random.shuffle(all_files)
            for fname in all_files:
                prev_file = next_file
                next_file = executor.submit(load_npy_file, fname, values)

                if prev_file:
                    output, input = prev_file.result()

                    # tweak batch size so we don't have a smaller batch left over
                    batches = input.shape[0] // BATCH_SIZE
                    actual_batch_size = input.shape[0] // batches + 1

                    for local_index in range(0, input.shape[0], actual_batch_size):
                        input_local = input[
                            local_index : (local_index + actual_batch_size)
                        ]
                        output_local = output[
                            local_index : (local_index + actual_batch_size)
                        ]

                        if hasattr(input_local, "todense"):
                            input_local = input_local.todense()

                        if hasattr(output_local, "todense"):
                            output_local = output_local.todense()

                        yield input_local, output_local


def build_model(hidden_layers=HIDDEN_LAYERS, *, output_layers, output_activation):
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

    def wdl_eval_mix(d):
        wdl = d.get("wdl") * 0.9
        evl = d.get("evl").reshape(-1, 1) * 0.1

        return wdl + evl

    train_generator = generate_npy_batches(files=train_files, values=wdl_eval_mix)

    if not model:
        model = build_model(output_layers=1, output_activation="tanh")
    else:
        model.summary()

    steps_per_epoch = int(len(train_files) * ENTRIES_PER_FILE / BATCH_SIZE)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=steps_per_epoch,
        decay_rate=1 - 1 / STATE_EPOCHS,
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss="mean_squared_error")

    mc = ModelCheckpoint(
        filepath="checkpoints/state."
        + str(INPUT_SIZE)
        + "x"
        + str(HIDDEN_LAYERS)
        + "x1.e{epoch:03d}-l{loss:.2f}.h5",
        verbose=True,
    )

    model.fit(
        train_generator,
        epochs=STATE_EPOCHS,
        verbose=1,
        callbacks=[mc],
        steps_per_epoch=steps_per_epoch,
        initial_epoch=start_epoch,
    )


# 0 - illegal movoe
# 1 - played move
# 2 - legal move
#
# therefore > 0.5 = all legal, > 1.5 = legal but not played
def correct_policy(target, output):
    output = tf.cast(output, tf.float32)

    move_is_legal = tf.greater_equal(target, 0.5)
    output = tf.where(move_is_legal, output, tf.zeros_like(output) - 1.0e10)

    legal_not_played = tf.greater_equal(target, 1.5)
    target = tf.where(legal_not_played, tf.zeros_like(target), target)

    return target, output


def policy_loss(target, output):
    target, output = correct_policy(target, output)

    policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(target), logits=output
    )

    return tf.reduce_mean(policy_cross_entropy)


def policy_acc(target, output):
    target, output = correct_policy(target, output)

    acc = tf.equal(tf.argmax(input=target, axis=1), tf.argmax(input=output, axis=1))

    return tf.reduce_mean(tf.cast(acc, tf.float32))


def train_policy(files, model, start_epoch):
    outputs = 384
    train_files = list(glob.glob(files))
    train_generator = generate_npy_batches(
        files=train_files, values=lambda d: d.get("moves")
    )

    if not model:
        model = keras.Sequential()
        model.add(keras.Input(shape=(INPUT_SIZE,)))
        model.add(
            layers.Dense(
                outputs,
                activation="linear",
                kernel_initializer="glorot_normal",
                use_bias=False,
                dtype=tf.float32,
            )
        )

    model.summary()

    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=policy_loss, metrics=[policy_acc])

    mc = ModelCheckpoint(
        filepath="checkpoints/policy."
        + str(INPUT_SIZE)
        + "x"
        + str(outputs)
        + ".e{epoch:03d}-l{loss:.2f}-a{policy_acc:.2f}.h5",
        verbose=True,
    )

    steps_per_epoch = int(len(train_files) * ENTRIES_PER_FILE / BATCH_SIZE)

    model.fit(
        train_generator,
        epochs=POLICY_EPOCHS,
        verbose=1,
        callbacks=[mc],
        steps_per_epoch=steps_per_epoch,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    train_what = sys.argv[1] if len(sys.argv) >= 2 else None
    model_file = sys.argv[2] if len(sys.argv) >= 3 else None
    start_epoch = int(sys.argv[3]) - 1 if len(sys.argv) >= 4 else 0

    model = None

    if model_file:
        model = keras.models.load_model(
            model_file,
            custom_objects=dict(policy_loss=policy_loss, policy_acc=policy_acc),
        )

    if train_what == "state":
        train_state("model_data/*.npy", model, start_epoch)
    elif train_what == "policy":
        train_policy("model_data/*.npy", model, start_epoch)
    else:
        print("Must specify to train either 'state' or 'policy'")
