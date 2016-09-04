# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training to train the network.
   This is a example for online triplet selection
   Each minibatch contains a set of archor-positive pairs, random select negative exemplar
"""

import caffe
import numpy as np
from numpy import *
import yaml
from sampledata import sampledata
import random
import cv2
from blob import prep_im_for_blob, im_list_to_blob
import config


class DataLayer(caffe.Layer):
    """Sample data layer used for training."""    
        
    
    def _get_next_minibatch(self):
        num_images = self._batch_size
        # Sample to use for each image in this batch
        sample = []
	personname_vec=[]
	while len(personname_vec)<self.num_c:
	    rand = random.randint(0,len(self.data_container._sample_person)-1)
            personname = self.data_container._sample_person.keys()[rand]
	    if ((personname) not in personname_vec) and len(self.data_container._sample_person[personname])>=self.num_im:
                personname_vec.append(personname)
	for person in personname_vec:
	    sample_per=[]
	    while len(sample_per)<self.num_im:
	        picindex = random.randint(0,len(self.data_container._sample_person[person])-1)
		if (self.data_container._sample_person[person][picindex]) not in sample_per:
                    sample_per.append(self.data_container._sample_person[person][picindex])
	    for i in sample_per:
		sample.append(i)
        im_blob,labels_blob = self._get_image_blob(sample)
        #print sample
        blobs = {'data': im_blob,
             'labels': labels_blob}
        return blobs

    def _get_image_blob(self,sample):
        im_blob = []
        labels_blob = []
        for i in range(self._batch_size):
            im = cv2.imread(config.IMAGEPATH+sample[i])
            personname = sample[i].split('/')[0]
            #print str(i)+':'+personname+','+str(len(sample))
            labels_blob.append(self.data_container._sample_label[personname])
            im = prep_im_for_blob(im)
            
            im_blob.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(im_blob)
        return blob,labels_blob

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)    
        self.num_c = config.num_class
	self.num_im=config.num_im
	self._batch_size=self.num_c*self.num_im
        self._name_to_top_map = {
            'data': 0,
            'labels': 1}

        self.data_container =  sampledata() 
        self._index = 0

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(self._batch_size, 1, 100, 100)

        top[1].reshape(self._batch_size)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            #top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

