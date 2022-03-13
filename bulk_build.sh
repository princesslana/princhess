#!/bin/bash

set -e
set -x

for e in $(seq -w 1 100)
do
  echo "Building princess-e$e..."
  cp train/state*e$e*.h5/* src/model
  cargo build --release
  cp target/release/princhess{,-e$e}
done
