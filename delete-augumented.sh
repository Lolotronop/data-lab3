#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="${1:-./ignored/data/training}"
PREFIX="${2:-aug_}"

if [ ! -d "$TRAIN_DIR" ]; then
  echo "Training directory not found: $TRAIN_DIR" >&2
  exit 1
fi

echo "Removing files starting with '$PREFIX' under $TRAIN_DIR"

find "$TRAIN_DIR" -type f -mindepth 2 -maxdepth 3 -name "${PREFIX}*" -print0 | \
  tee /dev/stderr | \
  xargs -0 --no-run-if-empty rm -v

