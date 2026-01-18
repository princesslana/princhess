#!/bin/bash

set -e

cd src

# Copy mounted config to avoid modifying the host file
cp config.yml config.yml.runtime

echo "token: $LICHESS_TOKEN" >> config.yml.runtime

python lichess-bot.py -v --config config.yml.runtime

