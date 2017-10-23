import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../python/')
import caffe

# Get the original network params
net = caffe.Net('deploy_original.prototxt', 1, weights='bvlc_alexnet.caffemodel')
#net = caffe.Net('deploy_original.prototxt', 'bvlc_alexnet.caffemodel', caffe.TEST)
# Read in the new network params
#net_new = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', caffe.TEST)
net_new = caffe.Net('deploy.prototxt', 1, weights='bvlc_alexnet.caffemodel')
# Modify the first layer weights to have 4 channels rather than 3
net_new.params['conv1_4c'][0].data[:,:3,:,:] = net.params['conv1'][0].data[...]

for c in range(0, 256):
    for y in range(0, 6):
        for x in range(0, 6): 
            net_new.params['fc6_n'][0].data[:,(y * 6 + x) * 256 + c] = net.params['fc6'][0].data[:,(c * 6 + y) * 6 + x]


# Save the new model
net_new.save('alexnet_four_channel_model.caffemodel')
