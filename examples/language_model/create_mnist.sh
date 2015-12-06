#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

EXAMPLE=examples/language_model
DATA=data/language_model
BUILD=build/examples/language_model

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/t_train_${BACKEND}
rm -rf $EXAMPLE/t_test_${BACKEND}

$BUILD/convert_data.bin $DATA/train.data.bin \
  $DATA/train.label.bin $EXAMPLE/t_train_${BACKEND} --backend=${BACKEND}

$BUILD/convert_data.bin $DATA/test.data.bin \
  $DATA/test.label.bin $EXAMPLE/t_test_${BACKEND} --backend=${BACKEND}

echo "Done."
