import numpy as np
import matplotlib.pyplot as plt

import sys

import caffe

# Get original model parameters
net = caffe.Net('deploy_original.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
# Get new model parameters
net_new = caffe.Net('deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
# Change weights in the first layer from 3 to 4 channels
net_new.params['conv1_1_4'][0].data[:,:3,:,:] = net.params['conv1_1'][0].data[...]
# Reorder the weights so that they map to HWCN layout
for c in range(0, 512):
    for y in range(0, 7):
        for x in range(0, 7): 
            net_new.params['fc6_n'][0].data[:,(y * 7 + x) * 512 + c] = net.params['fc6'][0].data[:,(c * 7 + y) * 7 + x]
#Save the new model
net_new.save('vgg_four_channel_model.caffemodel')
