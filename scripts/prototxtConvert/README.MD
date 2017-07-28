Assumptions  

1. Data layer batch size should be 192 to 256 and should be a multiple of 32.
2. Batch size for input data use nearest approximation to multiple of 32 between 192 and 256, so 100 will be 
   approximated to 192 rather than 256
3. After invert half converting layers, this program has only handled Accuracy and SoftmaxWithLoss layers seperatly becasue they have two bottom layers and inherite top names from before invert half converting layers
assume pad is necessary because most images consists of 3 channels and the model required pictures with 4 channels
norm and dropout layers are included as it was in the output file
num_cu parameter need to be set manually
