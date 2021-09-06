#!/bin/bash

set -e

PRINCHESS=${PRINCHESS:-../target/release/princhess}

for f in pgn/*.pgn
do
  echo "Featurizing $f..."

  $PRINCHESS -t $f

  echo "Splitting data..."

  split -l 1000000 train_data.libsvm model_data/$(basename $f).libsvm.

  rm train_data.libsvm
done




