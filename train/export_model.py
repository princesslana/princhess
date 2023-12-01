import numpy
import os
import sys
import tensorflow as tf
from numpy import array2string
from tensorflow import keras
from tensorflow.nn import softmax_cross_entropy_with_logits
from tensorflow.python.ops.numpy_ops import np_config

from training import policy_loss, policy_acc

np_config.enable_numpy_behavior()

QA = 256
QB = 256
QAB = QA * QB

def write_coefs(file, coefs):
    numpy.set_printoptions(threshold=numpy.inf)

    with open(file, "w") as f:
        print(array2string(coefs, separator=","), file=f)


def q(n, t):
    return tf.cast(t * n, dtype=tf.int32)


def export_state(file):
    print(f"Exporting from {file}...")
    model = keras.models.load_model(
        file,
    )

    output_folder = os.path.basename(file)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    (hidden_weights, hidden_bias) = model.layers[0].get_weights()
    write_coefs(os.path.join(output_folder, f"hidden_weights"), q(QA, hidden_weights))
    write_coefs(os.path.join(output_folder, f"hidden_bias"), q(QA, hidden_bias))

    (output_weights,) = model.layers[1].get_weights()
    write_coefs(
        os.path.join(output_folder, "output_weights"),
        numpy.transpose(q(QB, output_weights)),
    )


def export_policy(file):
    print(f"Exporting from {file}...")
    model = keras.models.load_model(
        file,
        custom_objects=dict(policy_loss=policy_loss, policy_acc=policy_acc),
    )

    output_folder = os.path.basename(file)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    (left_weights, left_bias) = model.layers[1].get_weights()
    write_coefs(os.path.join(output_folder, f"left_weights"), q(QA, left_weights))
    write_coefs(os.path.join(output_folder, f"left_bias"), q(QA, left_bias))

    (right_weights, right_bias) = model.layers[2].get_weights()
    write_coefs(os.path.join(output_folder, f"right_weights"), q(QA, right_weights))
    write_coefs(os.path.join(output_folder, f"right_bias"), q(QA, right_bias))

if __name__ == "__main__":
    export_what = sys.argv[1] if len(sys.argv) > 1 else None
    model_files = sys.argv[2:] if len(sys.argv) > 2 else None

    if export_what == "state":
        for model_file in model_files:
            export_state(model_file)
    elif export_what == "policy":
        for model_file in model_files:
            export_policy(model_file)
    else:
        print("Must specify to export either 'state' or 'policy'")
