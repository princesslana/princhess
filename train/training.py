import datetime
import glob
import numpy
import os
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
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.nn import softmax_cross_entropy_with_logits

# pieces + threats + defenses
INPUT_SIZE = 768 * 3;
HIDDEN_LAYERS = 384

STATE_EPOCHS = 40
POLICY_EPOCHS = 30

STEPS_PER_EPOCH = 10000

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

                    for local_index in range(0, input.shape[0], BATCH_SIZE):
                        input_local = input[local_index : (local_index + BATCH_SIZE)]
                        output_local = output[local_index : (local_index + BATCH_SIZE)]

                        if hasattr(input_local, "todense"):
                            input_local = input_local.todense()

                        if hasattr(output_local, "todense"):
                            output_local = output_local.todense()

                        if input_local.shape[0] == BATCH_SIZE:
                            yield input_local, output_local


def train_state(name, files, model, start_epoch):
    train_files = list(glob.glob(files))

    train_generator = generate_npy_batches(files=train_files, values=lambda d: d.get("wdl"))

    if not model:
        model = keras.Sequential()
        model.add(keras.Input(shape=(INPUT_SIZE,)))
        model.add(
            layers.Dense(
                HIDDEN_LAYERS,
                activation="relu",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                dtype=tf.float32,
            )
        )
        model.add(
            layers.Dense(
                1,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                use_bias=False,
                dtype=tf.float32,
            )
        )

    model.summary()

    lr_schedule = PiecewiseConstantDecay(
        values=[1e-3, 1e-4],
        boundaries=[STEPS_PER_EPOCH * STATE_EPOCHS // 2],
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss="mean_squared_error")

    if not name:
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    checkpoint_dir = "checkpoints/" + name
    log_dir = "logs/fit/" + name

    os.mkdir(checkpoint_dir)

    mc = ModelCheckpoint(
        filepath=checkpoint_dir
        + "/state."
        + str(INPUT_SIZE)
        + "x"
        + str(HIDDEN_LAYERS)
        + "x1.e{epoch:03d}-l{loss:.2f}.h5",
        verbose=True,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(
        train_generator,
        epochs=STATE_EPOCHS,
        verbose=1,
        callbacks=[mc, tensorboard_callback],
        steps_per_epoch=STEPS_PER_EPOCH,
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


def train_policy(name, files, model, start_epoch):
    outputs = 384
    train_files = list(glob.glob(files))
    train_generator = generate_npy_batches(
        files=train_files, values=lambda d: d.get("moves")
    )

    if not model:
        inp = keras.Input(shape=(INPUT_SIZE,))
        
        left = layers.Dense(
            outputs,
            activation="linear",
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
        )(inp)

        right = layers.Dense(
            outputs,
            activation="relu",
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
        )(inp)

        add = layers.Dense(
            outputs,
            activation="linear",
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
        )(inp)

        left_x_right = layers.Multiply()([left, right])

        output = layers.Add()([left_x_right, add])

        model = keras.Model(inputs=inp, outputs=output)

    model.summary()

    lr_schedule = PiecewiseConstantDecay(
        values=[1e-3, 1e-4],
        boundaries=[STEPS_PER_EPOCH * POLICY_EPOCHS // 2],
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss=policy_loss, metrics=[policy_acc])

    if not name:
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    checkpoint_dir = "checkpoints/" + name
    log_dir = "logs/fit/" + name

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    mc = ModelCheckpoint(
        filepath=checkpoint_dir
        + "/policy."
        + str(INPUT_SIZE)
        + "x"
        + str(outputs)
        + ".e{epoch:03d}-l{loss:.2f}-a{policy_acc:.2f}.h5",
        verbose=True,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(
        train_generator,
        epochs=POLICY_EPOCHS,
        verbose=1,
        callbacks=[mc, tensorboard_callback],
        steps_per_epoch=STEPS_PER_EPOCH,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    train_what = sys.argv[1] if len(sys.argv) >= 2 else None
    name = sys.argv[2] if len(sys.argv) >= 3 else None
    model_file = sys.argv[3] if len(sys.argv) >= 4 else None
    start_epoch = int(sys.argv[4]) - 1 if len(sys.argv) >= 5 else 0

    model = None

    if model_file:
        model = keras.models.load_model(
            model_file,
            custom_objects=dict(policy_loss=policy_loss, policy_acc=policy_acc),
        )

    if train_what == "state":
        train_state(name, "model_data/*.npy", model, start_epoch)
    elif train_what == "policy":
        train_policy(name, "model_data/*.npy", model, start_epoch)
    else:
        print("Must specify to train either 'state' or 'policy'")
