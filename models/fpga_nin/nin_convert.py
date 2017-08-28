import numpy as np
import matplotlib.pyplot as plt

import sys

import caffe

# Get the original network params
net = caffe.Net('deploy_original.prototxt', 'nin_imagenet_conv.caffemodel', caffe.TEST)
# Read in the new network params
net_new = caffe.Net('deploy.prototxt', 'nin_imagenet_conv.caffemodel', caffe.TEST)
# Modify the first layer weights to have 4 channels rather than 3
net_new.params['conv1_4c'][0].data[:,:3,:,:] = net.params['conv1'][0].data[...]
# Save the new model
net_new.save('nin_four_channel_model.caffemodel')
