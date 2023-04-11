#!/bin/bash

set -e

pgn-extract -#500000 -C -N -V -s --nobadresults *.pgn.in

