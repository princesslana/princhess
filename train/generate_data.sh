#!/bin/bash

set -e

PRINCHESS=${PRINCHESS:-../target/release/princhess}

state_data() {
  echo "Sampling state data..."
  for f in pgn/*.pgn
  do
    echo "Featurizing $f..."

    $PRINCHESS -t $f

    echo "Splitting data..."

    rm -f model_data/*.libsvm.*
    split -l 1000000 train_data.libsvm model_data/$(basename $f).libsvm.

    rm train_data.libsvm
    rm -f model_data/*.gz
  done
}

policy_data() {
  echo "Sampling policy data..."
  for f in pgn/*.pgn
  do
    echo "Featurizing $f..."

    $PRINCHESS -p -t $f

    echo "Splitting data..."

    rm -f policy_data/*.libsvm.*
    split -l 1000000 policy_train_data.libsvm policy_data/$(basename $f).libsvm.

    rm policy_train_data.libsvm
    rm -f policy_data/*.gz
  done
}

case $1 in
  "state")
    state_data
    ;;
  "policy")
    policy_data
    ;;
  *)
    echo "Must specify either 'state' or 'policy'"
    ;;
esac

