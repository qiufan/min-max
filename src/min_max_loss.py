# --------------------------------------------------------
# min-max LOSS
# Copyright (c) 2016 qiufan Tech.
# Shi W, Gong Y, Wang J. Improving CNN Performance with Min-Max Objective[J].
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

class MinMaxLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the MinMaxLossLayer."""
        assert shape(bottom[0].data) == shape(bottom[1].data)

	layer_params = yaml.load(self.param_str)
        self.k1 = layer_params['k1']
	self.k2 = layer_params['k2']
	self.sigma_2 = layer_params['sigma']

        top[0].reshape(1)

    def forward(self, bottom, top):
	size=bottom[0].num
	dis_matrix=np.zeros((size,size))
	Tk1=np.zeros((size,size))
	Tk2=np.zeros((size,size))
	for i in range(size):
	    for j in range(size):
		if i==j:
		    continue
		x1=bottom[0].data[i]
		x2=bottom[0].data[j]
		x1_x2=x1-x2
		dis=np.dot(x1_x2,x1_x2)
		dis_matrix(i,j)=dis
	for i in range(size):
	    k1_nearest={}
	    k2_nearest={}
	    for j in range(size):
		if i==j:
		    continue
		if bottom[1].data[i]==bottom[1].data[j]:
		    k1_nearest[j]=dis_matrix[i][j]
		if bottom[1].data[i]!=bottom[1].data[j]:
		    k2_nearest[j]=dis_matrix[i][j]
	    k1_nearest = sorted(k1_nearest.items(), key = lambda d: d[1], reverse = True)
	    print k1_nearest
	    k2_nearest = sorted(k2_nearest.items(), key = lambda d: d[1], reverse = True)
	    for s in range(self.k1):
		Tk1(i,k1_nearest[s][0])=1
	    for s in range(self.k2):
		Tk2(i,k2_nearest[s][0])=1
	Loss=0
	Loss1=0
	Loss2=0
	for i in range(size):
	    for j in range(size):
		if i==j:
		    continue
		if Tk1(i,j)==1 or Tk1(j,i)==1:
		    Loss1+=dis_matrix(i,j)
		if Tk2(i,j)==1 or Tk2(j,i)==1:
		    Loss2+=dis_matrix(i,j)
	Loss=Loss1-Loss2
	Loss=Loss/bottom[0].num
	top[0].data[...] = Loss
	
        
    def backward(self, top, propagate_down, bottom):
        
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
