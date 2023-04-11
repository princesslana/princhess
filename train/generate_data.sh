#!/bin/bash

set -e

PRINCHESS=${PRINCHESS:-../target/release/princhess}

state_data() {
  echo "Sampling state data..."

  rm -f model_data/*.libsvm.*

  for f in pgn/*.pgn
  do
    echo "Featurizing $f..."

    $PRINCHESS -t $f -o $f.libsvm &
  done

  wait

  for f in pgn/*.pgn
  do
    echo "Calculating split..."

    samples=$(wc -l < $f.libsvm)
    splits=$(( $samples / 1000000 ))
    split_size=$(( $samples / $splits + 1))

    echo "Splitting data ($split_size)..."

    split -l $split_size $f.libsvm model_data/$(basename $f).libsvm.

    rm $f.libsvm
    rm -f model_data/*.gz
  done
}

policy_data() {
  echo "Sampling policy data..."

  rm -f policy_data/*.libsvm.*

  for f in pgn/*.pgn
  do
    echo "Featurizing $f..."

    $PRINCHESS -p -t $f

    echo "Calculating split..."

    samples=$(wc -l < policy_train_data.libsvm)
    splits=$(( $samples / 1000000 ))
    split_size=$(( $samples / $splits + 1))

    echo "Splitting data ($split_size)..."

    mkdir -p policy_data
    split -l $split_size policy_train_data.libsvm policy_data/$(basename $f).libsvm.

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
