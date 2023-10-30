#!/bin/bash

set -e

PRINCHESS=${PRINCHESS:-../target/release/princhess}
SYZYGY_PATH=${SYZYGY_PATH:-../syzygy}
FILES=${FILES:-pgn/*.pgn}

generate_data() {
  echo "Sampling data..."

  ls -S $FILES | parallel -u -j 6 $PRINCHESS -t {} -o {}.libsvm -s $SYZYGY_PATH

  rm -f model_data/*.libsvm.*

  for f in $FILES
  do
    echo "Calculating split..."

    samples=$(wc -l < $f.libsvm)
    splits=$(( $samples / 1000000 ))
    split_size=$(( $samples / $splits + 1))

    echo "Splitting data ($split_size)..."

    split -l $split_size $f.libsvm model_data/$(basename $f).libsvm.

    rm -f model_data/*.gz
  done
}

generate_data
