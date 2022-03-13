import numpy
import os
import sys
from numpy import array2string
from tensorflow import keras


def write_coefs(file, coefs):
    numpy.set_printoptions(threshold=numpy.inf)

    with open(file, "w") as f:
        print(array2string(coefs, separator=","), file=f)


def model_to_coefs(file):
    print(f"Exporting from {file}...")
    model = keras.models.load_model(file)

    hidden_weights, hidden_bias = model.layers[0].get_weights()
    (output_weights,) = model.layers[1].get_weights()

    output_folder = os.path.basename(file)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    write_coefs(
        os.path.join(output_folder, "hidden_weights"), numpy.transpose(hidden_weights)
    )
    write_coefs(os.path.join(output_folder, "hidden_bias"), hidden_bias)
    write_coefs(
        os.path.join(output_folder, "output_weights"), numpy.transpose(output_weights)
    )

for f in sys.argv[1:]:
    model_to_coefs(f)
