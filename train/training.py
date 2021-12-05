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

    split_idx = max(1, len(all_files) // 10)
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

            output = numpy.where(output==0, 2, output)
            output = numpy.where(output==1, 0, output)
            output = numpy.where(output==-1, 1, output)

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

        file_y = numpy.where(file_y==0, 2, file_y)
        file_y = numpy.where(file_y==1, 0, file_y)
        file_y = numpy.where(file_y==-1, 1, file_y)

        all_x = vstack((all_x, file_x)) if all_x is not None else file_x
        all_y = numpy.concatenate((all_y, file_y)) if all_y is not None else file_y

    print(f"{len(all_y)} samples loaded.")
    return (all_x.todense(), all_y)

def model_builder(hp):
    decimals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    hp_hidden_layer = hp.Choice('hidden_layer', values=[32, 64, 128, 256, 512, 1024])
    hp_learning_rate = hp.Choice("learning_rate", values=decimals)
    #hp_decay_steps = hp.Choice("decay_steps", values=[1e4, 1e5, 1e6, 1e7, 1e8])
    #hp_decay_rate = hp.Float('decay_rate', min_value=0.5, max_value=1.0)
    hp_init = hp.Choice('initializer', values=['zeros', 'ones'])
    hp_output_bias = hp.Boolean("output_bias")
    #hp_l1 = hp.Choice("l1", values=[0.0] + decimals)
    #hp_l2 = hp.Choice("l2", values=[0.0] + decimals)
    #hp_beta_1 = hp.Float('beta_1', min_value=0.75, max_value=1.0)
    #hp_beta_2 = hp.Float('beta_2', min_value=0.9, max_value=1.0)


    model = keras.Sequential()
    model.add(keras.Input(shape=(768,)))
    model.add(layers.Dense(hp_hidden_layer, activation="relu", kernel_initializer=hp_init))
                           #kernel_regularizer=keras.regularizers.l1_l2(l1=hp_l1, l2=hp_l2)))
    model.add(layers.Dense(3, activation="softmax", kernel_initializer=hp_init, use_bias=hp_output_bias))
    model.summary()

    #lr = keras.optimizers.schedules.ExponentialDecay(hp_learning_rate,
    #                                                 decay_steps=hp_decay_steps,
    #                                                 decay_rate=hp_decay_rate)
    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    #optimizer = keras.optimizers.SGD(learning_rate=lr)

    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"]
    )

    return model

def tune_with_keras(files):
    train_files, test_files = split_files_train_and_test(list(glob.glob(files)))
    batch_size = 256
    test_data = load_files(test_files)
    train_generator = generate_batches(files=train_files, batch_size=batch_size)

    tuner = kt.Hyperband(model_builder, objective="val_loss", max_epochs=50, factor=3, project_name="state_tuning")

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    tuner.search(train_generator, epochs=200, callbacks=[es], verbose=1,
                 validation_data = test_data,
                 steps_per_epoch=len(train_files) * 1000000 / batch_size / 2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best:")
    for param in ['hidden_layer', 'learning_rate', 'initializer', 'output_bias']:
        print(f"{param}: {best_hps.get(param)}")


def train_with_keras(files):
    train_files, test_files = split_files_train_and_test(list(glob.glob(files)))
    batch_size = 256
    test_data = load_files(test_files)
    train_generator = generate_batches(files=train_files, batch_size=batch_size)

    hidden_layers=32

    model = keras.Sequential()
    model.add(keras.Input(shape=(768,)))
    model.add(layers.Dense(hidden_layer, activation="relu", kernel_initializer="he_normal"))
    model.add(layers.Dense(1, activation="tanh", kernel_initializer="he_normal", use_bias=False))
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="mean_squared_error")

    mc = ModelCheckpoint(
        filepath="checkpoints/state.768x" + str(hidden_layers) + "e{epoch:03d}-l{val_loss:.2f}.h5",
        verbose=True,
    )

    model.fit(
        train_generator,
        epochs=500,
        verbose=1,
        callbacks=[mc],
        steps_per_epoch=len(train_files) * 1000000 / batch_size,
        validation_data=test_data
    )


def train(classifier, files):
    train_files = list(glob.glob(files))

    print(f"Training from {len(train_files)} files...")

    for iteration in range(5):
        print("Fitting...")
        random.shuffle(train_files)
        # classifier.set_params(learning_rate_init=learning_rate)
        for i, f in enumerate(train_files):
            print(
                f"Loading {f} ({i + 1}/{len(train_files)}, iteration {iteration + 1})..."
            )
            x, y = load_svmlight_file(f)
            print(f"{len(y)} samples loaded")

            y_shuffled, x_shuffled = sklearn.utils.shuffle(y, x)

            classifier.partial_fit(x, y)

    return classifier.coefs_, classifier.intercepts_


def write_coefs(file, coefs):
    numpy.set_printoptions(threshold=numpy.inf)

    with open(file, "w") as f:
        print(array2string(coefs, separator=","), file=f)


def model_to_coefs(file):
    model = keras.models.load_model(file)

    hidden_weights, hidden_bias = model.layers[0].get_weights()
    output_weights, output_bias = model.layers[1].get_weights()

    write_coefs("hidden_weights", numpy.transpose(hidden_weights))
    write_coefs("hidden_bias", hidden_bias)
    write_coefs("output_weights", numpy.transpose(output_weights))
    write_coefs("output_bias", output_bias)


def train_state():
    #tune_with_keras(files="model_data/*.libsvm.*")
    train_with_keras(files="model_data/*.libsvm.*")


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
