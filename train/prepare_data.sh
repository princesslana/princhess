#!/bin/bash

set -e

PREPARE_DATA_PY=${PREPARE_DATA_PY:-prepare_data.py}
FILES=${FILES:-model_data/*.libsvm.*}

ls $FILES | parallel -j 6 python prepare_data.py {}
