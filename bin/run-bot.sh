#!/bin/bash

set -e

cd src

echo "token: $LICHESS_TOKEN" >> config.yml

python lichess-bot.py

