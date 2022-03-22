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
from sklearn.datasets import load_svmlight_file
from tensorflow import keras
from tensorflow.keras import activations, layers, regularizers
from tensorflow.nn import softmax_cross_entropy_with_logits

# pieces + last capture + threats
INPUT_SIZE = 768 + (5 * 64) + 768

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
                            output_local, input_local = numpy.hsplit(
                                input_local.todense(), [categories]
                            )
                            yield input_local, output_local
                            # output_local = input_local[0:categories, :].todense()
                            # nput_local = input_local[categories:, :]

                            # output_local = keras.utils.to_categorical(
                            #    output_local, num_classes=categories
                            # )

                        else:
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


def square_sigmoid(inp):
    # Equalivalent to 1 if inp > 0.5 else 0
    half = tf.constant(0.5, inp.dtype.base_dtype)
    return tf.keras.activations.relu(tf.add(inp, half), max_value=1.0, threshold=1.0, alpha=0.0)

    # return tf.where(tf.less_equal(inp, 0), tf.zeros_like(inp), tf.ones_like(inp))


# 0 - illegal movoe
# 1 - played move
# 2 - legal, not played move
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
    outputs = 1792
    train_files = list(glob.glob(files))
    train_generator = generate_batches(files=train_files, categories=outputs)

    if not model:
        model = keras.Sequential()
        model.add(keras.Input(shape=(INPUT_SIZE,)))
        model.add(
            layers.Dense(
                outputs,
                activation="linear",
                kernel_initializer="glorot_normal",
                # kernel_regularizer=regularizers.L1(0.0001),
                use_bias=False,
                dtype=tf.float32,
            )
        )

        """
        inputs = keras.Input(shape=(INPUT_SIZE,))
        hiddens = layers.Dense(hidden, activation=square_sigmoid, use_bias=True, dtype=tf.float32,
            kernel_initializer="he_normal")(inputs)

        merged = layers.Concatenate()([hiddens, inputs])

        outputs = layers.Dense(
                outputs,
                activation="linear",
                kernel_initializer="glorot_normal",
                # kernel_regularizer=regularizers.L1(0.0001),
                use_bias=False,
                dtype=tf.float32,
            )(merged)

        model = keras.Model(inputs, outputs=outputs)
        """

    model.summary()

    """
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = [10, 55], values = [1e-2, 1e-3, 1e-4])
    """

    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=1000*10,
        decay_rate=0.5,
        staircase=True,
    )
    """

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=policy_loss, metrics=[policy_acc])

    mc = ModelCheckpoint(
        filepath="checkpoints/policy."
        + str(INPUT_SIZE)
        + "x"
        + str(outputs)
        + ".e{epoch:03d}-l{loss:.2f}-a{policy_acc:.2f}.h5",
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


if __name__ == "__main__":
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
