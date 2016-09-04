# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""


import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import math

class Norm2Layer(caffe.Layer):
    """norm2 layer used for L2 normalization."""

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        
        top[0].reshape(bottom[0].num, shape(bottom[0].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        minibatch_db = []
        
        for i in range((bottom[0]).num):
            X_normalized = preprocessing.normalize(bottom[0].data[i].reshape(1,-1), norm='l2')[0]
            minibatch_db.append(X_normalized)
        #print 'bottom**:',np.dot(bottom[0].data[0],bottom[0].data[0])
        top[0].data[...] = minibatch_db

    def backward(self, top, propagate_down, bottom):
        #"""This layer does not need to backward propogate gradient"""
        #pass
	top_minibatch_diff_db=[]
	top_minibatch_data_db=[]
	bottom_minibatch_data_db=[]

	for i in range((bottom[0]).num):
	    top_minibatch_diff_db.append(top[0].diff[i])
	    top_minibatch_data_db.append(top[0].data[i])
	    bottom_minibatch_data_db.append(bottom[0].data[i])
	for i in range((bottom[0]).num):
	    mul_y_diff_data=np.dot(top_minibatch_diff_db[i],top_minibatch_data_db[i])
	    sqar_mul_xt_x=math.sqrt(np.dot(bottom_minibatch_data_db[i],bottom_minibatch_data_db[i]))
	    bottom[0].diff[i]=(top_minibatch_diff_db[i]-mul_y_diff_data*top_minibatch_data_db[i])*1.0/sqar_mul_xt_x
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
