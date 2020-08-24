#!/usr/bin/env bash
set -e

python build.py
cat .build/script.py
