# -*- coding: utf-8 -*-
#生成类似mnist的数据
"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

import sys
import lmdb
import random
import subprocess
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
from numpy import array
warnings.filterwarnings("ignore")   

sys.path.append('python/caffe/proto'); import caffe_pb2
from caffe_pb2 import Datum

DEF_IDX=int(0)
W2V_LEN=int(300)

def write_to_file(fp, type, data_list):
    out = array(data_list, type)
    out.tofile(fp)

def create_lmdb_file(dataset, phase, w2v_dict, height, width):

    print 'Starting %s' % phase
    data_filename = './data/language_model/%s.data.bin' % phase
    subprocess.call(['rm', '-rf', data_filename])
    label_filename = './data/language_model/%s.label.bin' % phase
    subprocess.call(['rm', '-rf', label_filename])

    data_nums= len(dataset)
    print 'Writing %s sentences, %s' % (data_nums, phase)

    fp_data = open(data_filename, "wb")
    fp_label = open(label_filename, "wb")

    #data file header
    magic_num=2051
    write_to_file(fp_data, 'int32', [magic_num])
    write_to_file(fp_data, 'int32', [data_nums])
    write_to_file(fp_data, 'int32', [height])
    write_to_file(fp_data, 'int32', [width])

    #label file header
    magic_num=2049
    write_to_file(fp_label, 'int32', [magic_num])
    write_to_file(fp_label, 'int32', [data_nums])

    index = 0
    w2v_dict_len = len(w2v_dict)
    for i in range(len(dataset)):
        sentence = dataset[i]
        y = sentence[-1]

        current = len(sentence) - 1
        if height != current:
            print sys._getframe().f_lineno, "length not equal"
            sys.exit(-1)

        float_data_list = []

        for j in range(0, len(sentence)-1):
            word_idx = sentence[j]
            elem_vector = []
            if word_idx < w2v_dict_len:
                elem_vector = w2v_dict[word_idx]
            else:
                elem_vector = w2v_dict[DEF_IDX]
                
            if len(elem_vector) != W2V_LEN:
                print sys._getframe().f_lineno, "w2v length not equal 300"
                sys.exit(-1)

            float_data_list.extend(list(elem_vector))

        ##data
        res = array(float_data_list,'float32')
        res.tofile(fp_data)

        '''
        res.shape = height, width
        for i in range(height):
            print res[i]
        '''

        ###label
        write_to_file(fp_label, 'int32', [y])

        index += 1
        if index % 100 == 0:
            print "finished num:", index 
            sys.stdout.flush()

    fp_data.close()
    fp_label.close()

    print 'Writing %s sentences, %s. End' % (len(dataset), phase)

def create_datasets(train, test, w2v, height) :
    print "create train lmdb"
    width = len(w2v[0])
    create_lmdb_file(train, "train", w2v, height, width)

    print "create test lmdb"
    create_lmdb_file(test, "test", w2v, height, width)

#### 保证句子长度为max_l+2*pad
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """

    x = []
    pad = filter_h - 1

    for i in xrange(pad):
        x.append(0)

    words = sent.split()

    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    while len(x) < max_l+2*pad:
        x.append(0)

    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """

    train, test = [], []

    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])

        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   

    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     
  
def test_print(W, word_idx_map, vocab):
    fp = open("word.txt", "w");
    for elem in word_idx_map:
        line = "%s:%s\n" % (elem, word_idx_map[elem])
        fp.write(line) 
    fp.close()
        
    fp = open("vocab.txt", "w");
    for elem in vocab:
        line = "%s\n" % (elem)
        fp.write(line) 
    fp.close()

    fp = open("w2v.txt", "w");
    for i, val in enumerate(W):
        line = ""
        target = ""
        for elem in val:
            target = "%s %s" % (target, elem)
        line = "%s\t%s\n" % (i, target)
        fp.write(line) 
    fp.close()

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: {0} w2v_file".format(sys.argv[0]))
        exit(1)

    print "loading data...",
    x = cPickle.load(open(sys.argv[1],"rb"))
    revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    print "data loaded!"

    random.shuffle(revs)

    ##句子有word编号构成
    [train, test] = make_idx_data_cv(revs, word_idx_map, 0, max_l=56, k=300, filter_h=5)
    max = 56+2*(5-1)
    print len(W)
    create_datasets(train, test, W, max)
