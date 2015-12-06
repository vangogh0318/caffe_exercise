# -*- coding: utf-8 -*-
#读类似mnist的数据
import sys
import numpy as np
import struct
from numpy import array

with open(sys.argv[1], "rb") as f:

    magic_num = np.fromstring(f.read(np.dtype('int32').itemsize), dtype='int32') 
    data_nums = np.fromstring(f.read(np.dtype('int32').itemsize), dtype='int32') 
    height = np.fromstring(f.read(np.dtype('int32').itemsize), dtype='int32') 
    heigth = int(height)
    width = np.fromstring(f.read(np.dtype('int32').itemsize), dtype='int32') 
    width = int(width)
    print "magic_num:", magic_num
    print "data_nums:", data_nums
    print "height:", height
    print "width:", width

    len = height * width
    print len
    binary_len = np.dtype('float32').itemsize * len
    buffer = np.fromstring(f.read(binary_len), dtype='float32') 
    buffer.shape = height, width
    for i in range(height):
        print buffer[i]
