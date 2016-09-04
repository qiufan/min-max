# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im):
    target_size = 128
    pixel_means = np.load('0607_mean.npy')
    channel_swap = (2,0,1)
    image_mean=np.zeros((pixel_means.shape[1],pixel_means.shape[2],3))
    for i in range(3):
	image_mean[:,:,i]=pixel_means[i,:,:]
    im = im.astype(np.float32, copy=False)
    image_mean = cv2.resize(image_mean, (target_size,target_size),
                    interpolation=cv2.INTER_LINEAR)
    im -= image_mean
    

    return im
