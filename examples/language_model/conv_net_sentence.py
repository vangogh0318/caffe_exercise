# -*- coding: utf-8 -*-

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
warnings.filterwarnings("ignore")   

sys.path.append('python/caffe/proto'); import caffe_pb2
from caffe_pb2 import Datum

DEF_IDX=int(0)
W2V_LEN=int(300)

def create_lmdb_file(dataset, phase, w2v_dict ):

    print 'Starting %s' % phase
    db_name = './examples/language_model/lm_%s_db' % phase
    subprocess.call(['rm', '-rf', db_name])
    env = lmdb.open(db_name, map_size=2147483648*8)

    print 'Writing %s sentences, %s' % (len(dataset), phase)

    last = None
    index = 0
    with env.begin(write=True) as txn:
        for i in range(len(dataset)):
            sentence = dataset[i]

            ##sentence
            datum = Datum()
            datum.channels = 1
            datum.width = 300
            datum.height = len(sentence)
            datum.label = int(sentence[-1])

            current = datum.height
            if last and last != current:
                print sys._getframe().f_lineno, "length not equal"
                sys.exit(-1)

            #print sys._getframe().f_lineno, "sentence length:", len(sentence)
            for j in range(0, len(sentence)-1):
                word_idx = sentence[j]
                #print word_idx

                if word_idx in w2v_dict:
                    elem_vector = w2v_dict[word_idx]
                else:
                    elem_vector = w2v_dict[DEF_IDX]
                    
                if len(elem_vector) != W2V_LEN:
                    print sys._getframe().f_lineno, "w2v length not equal 300"
                    sys.exit(-1)

                for elem in elem_vector:
                    datum.float_data.append(elem)
                #datum.float_data += elem_vector


            key = str(i)
            txn.put(key, datum.SerializeToString())

            index += 1

            if index % 100 == 0:
                print "finished num:", index 
                sys.stdout.flush()

            '''
            if index > 2 :
                print '11111111111111111111111111111111111111111111'
                break;
            '''
            last = current

    print 'Writing %s sentences, %s. End' % (len(dataset), phase)

def create_datasets(train, test, w2v) :
    print "create train lmdb"
    create_lmdb_file(train, "train", w2v)

    print "create test lmdb"
    create_lmdb_file(test, "test", w2v)

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

    ##句子有word编号构成
    [train, test] = make_idx_data_cv(revs, word_idx_map, 0, max_l=56,k=300, filter_h=5)
    create_datasets(train, test, W)
