Assumptions:  

1. Data layer batch size should be 192 to 256 and should be a multiple of 32
2. Data layer batch size for input data use nearest approximation to multiple of 32 between 192 and 256, so 100 will be 
   approximated to 192 rather than 256
3. Input channel should be multiple of 4, if not, add pad layer to the front
4. Assume pad is necessary because most images consists of 3 channels and the model required pictures with 4 channels
5. Convolution layer need to have stride of 1, input channels should be multiple of 4 and output channels needs to be
   multiple of 16
6. Pooling layer needs to be with stride equals to 2 and pooling window of 2 or 3
7. Pooling layer window size can be value other than 2 or 3 if it's near the end of the network (e.g. right before the loss        layers)
8. Inner product number of outputs need to be a multiple of 16 except for the last inner product layer
9. After the invert half converting layers, this program has only handled Accuracy and SoftmaxWithLoss layers seperatly 
   becasue they have two bottom layers and inherite top names from before invert half converting layers
10. Norm and dropout layers are included as it was in the output file
11. num_cu parameter needs to be set manually

Output is in FLGA_Caffe.prototxt
Two example input and output files are presented
test1_original.txt and test2_original.txt are the original Caffe model files
test1_output.txt and test2_output.txt are the output files after convertion

end_program.txt setup_pad_layers.txt setup_noPad.txt are necessary files used to convert any prototxt
