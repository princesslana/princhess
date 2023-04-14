import numpy
import os
import sys
from numpy import array2string
from tensorflow import keras
from tensorflow.nn import softmax_cross_entropy_with_logits

from training import policy_loss, policy_acc, square_sigmoid


def write_coefs(file, coefs):
    numpy.set_printoptions(threshold=numpy.inf)

    with open(file, "w") as f:
        print(array2string(coefs, separator=","), file=f)


def model_to_coefs(file):
    print(f"Exporting from {file}...")
    model = keras.models.load_model(
        file,
        custom_objects=dict(
            policy_loss=policy_loss, policy_acc=policy_acc, square_sigmoid=square_sigmoid
        ),
    )

    output_folder = os.path.basename(file)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    idx = 0

    while idx <  len(model.layers) - 1:
        if isinstance(model.layers[idx], keras.layers.Dense):
            hidden_weights, hidden_bias = model.layers[idx].get_weights()
            write_coefs(
                os.path.join(output_folder, f"hidden_weights_{idx}"), hidden_weights
            )
            write_coefs(os.path.join(output_folder, f"hidden_bias_{idx}"), hidden_bias)

        idx += 1

    (output_weights,) = model.layers[idx].get_weights()
    write_coefs(
        os.path.join(output_folder, "output_weights"), numpy.transpose(output_weights)
    )


for f in sys.argv[1:]:
    model_to_coefs(f)
