import sys
import lmdb
import numpy as np
import os

sys.path.append('python/caffe/proto'); import caffe_pb2
from caffe_pb2 import Datum

lmdb_env = lmdb.open(sys.argv[1])
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

datum = caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    height = datum.height
    width = datum.width

    nparray = np.array(datum.float_data)
    nparray.shape = height, width
    print "idx:", key, label, height, width
    for i in range(height) :
        print nparray[i]
