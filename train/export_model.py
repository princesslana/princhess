import numpy
import sys
from numpy import array2string
from tensorflow import keras

def write_coefs(file, coefs):
    numpy.set_printoptions(threshold=numpy.inf)

    with open(file, "w") as f:
        print(array2string(coefs, separator=","), file=f)


def model_to_coefs(file):
    model = keras.models.load_model(file)

    hidden_weights, hidden_bias = model.layers[0].get_weights()
    output_weights, = model.layers[1].get_weights()

    write_coefs("hidden_weights", numpy.transpose(hidden_weights))
    write_coefs("hidden_bias", hidden_bias)
    write_coefs("output_weights", numpy.transpose(output_weights))

model_to_coefs(sys.argv[1])
