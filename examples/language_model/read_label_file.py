# -*- coding: utf-8 -*-
#读类似mnist的label数据
import sys
import numpy as np
import struct
from numpy import array

with open(sys.argv[1], "rb") as f:

    magic_num = np.fromstring(f.read(np.dtype('int32').itemsize), dtype='int32') 
    data_nums = np.fromstring(f.read(np.dtype('int32').itemsize), dtype='int32') 
    print "magic_num:", magic_num
    print "data_nums:", data_nums

    for i in range(data_nums):
        binary_len = np.dtype('int32').itemsize
        buffer = np.fromstring(f.read(binary_len), dtype='int32') 
        print buffer
