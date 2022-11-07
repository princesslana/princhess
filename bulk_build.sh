#!/bin/bash

set -e
set -x

case $1 in
  state)
    what=state
    where=src/model
    ;;
  policy)
    what=
    where=src/policy
    ;;
  *)
    echo "Must specify state or policy"
    exit
    ;;
esac

for e in $(seq -w 001 100)
do
  echo "Building princess-e$e..."
  cp train/$what*e$e*.h5/* $where
  cargo build --release
  cp target/release/princhess{,-e$e}
done
