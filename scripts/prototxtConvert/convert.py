import subprocess
import os
from shutil import copyfile
import numpy as np
import fileinput
import re

# Assumptions
# batch size for input data use nearest approximation, so 100 will be approximated to 192 rather than 256
# after invert half converting layers, this program has only handled Accuracy and SoftmaxWithLoss layers seperatly becasue they have two bottom layers and inherite top names from before invert half converting layers
# assume pad is necessary because most images consists of 3 channels and the model required pictures with 4 channels
# norm and dropout layers are included as it was in the output file
# num_cu parameter need to be set manually


def copyFromLinetoLine(start_line, end_line, original, destination, mode):
	des = open(destination, mode)
	with open(original) as myFile:
		for num, line in enumerate(myFile, 1):
			if (num >= start_line and num <= end_line):
				des.write(line)
	myFile.close()

def round_up(num, divisor):
    return num + (16-(num%divisor))

def batch_size_processor(batch_size):
	if (batch_size >= 192) and (batch_size <= 256):
		if (batch_size % 32 == 0):
			return batch_size
		else:
			return 256
	# assume rounding to nearest value between 192 and 256
	elif (batch_size < 192):
		return 192
	else:
		return 256

# conv - pooling - relu detector
def pooling_relu_detector (directory, line_number):
	thisFile = 'temp.txt'
	copyfile(directory, thisFile)
	raw_file = open('temp.txt', 'r')
	raw_text = raw_file.readlines()
	layer_counter = 0
	# if the layer after the next layer is relu, return true
	for current_line in range(line_number, len(raw_text)):
		line = raw_text[current_line]
		if 'type' in line:
			layer_counter += 1
		if (layer_counter == 2):
			if 'type: "ReLU"' in line:
				return True
			else:
				return False

directory = '/home/lin/Desktop/prototxt/lenet_train_test_2.prototxt'
destination = '/home/lin/Desktop/prototxt/FPGA.txt'
program_pad = '/home/lin/Desktop/prototxt/setup_pad_layers.txt'
program_noPad = '/home/lin/Desktop/prototxt/setup_noPad.txt'
end_program_layers = '/home/lin/Desktop/prototxt/end_program.txt'
output_file = '/home/lin/Desktop/prototxt/FPGA_Caffe.prototxt'

destination_append_mode = open(destination, 'a+')
thisFile = 'temp.txt'
copyfile(directory, thisFile)

bracket_count = -1
end_line = -1

#block identification
data_block = False
convolution_block = False
pooling_block = False
inner_product_block = False
ReLU_block = False
implementation_input_block = False
norm_block = False
drop_out_block = False
dropout_block = False

inner_pooling_block = False
inner_conv_block = False
inner_inner_product_block = False

finish_program = False
finish_program_layer = False
finish_end_program = False
set_ending_layer_start = False
num_data_layer = 0
last_inner_product_num_output = 0

raw_file = open('temp.txt', 'r')
raw_text = raw_file.readlines()

with open('temp.txt') as myFile:
	print 'Processing layers...'
	for line_number, line in enumerate(myFile, 1):
		
		if (line_number == 1):
			# copy first line, the name of the model and cleans the old contents
			copyFromLinetoLine(1, 1, directory, destination, 'w')

		if 'layer {' in line:
			bracket_count = 1
			# keep track of the starting line number of each block
			current_block_start_line = line_number
		elif '{' in line:
			bracket_count += line.count('{')
			bracket_count -= line.count('}')
		elif '}' in line:
			bracket_count -= 1

		# flag settings for block identification
		if 'type: "Data"' in line:
			if (bracket_count != 0):
				data_block = True
				convolution_block = False
				pooling_block = False
				inner_product_block = False
				ReLU_block = False
				implementation_input_block = False
				norm_block = False
				dropout_block = False
				start_data = line_number - 2

		elif 'type: "Convolution"' in line:
			if (bracket_count != 0):
				data_block = False
				convolution_block = True
				pooling_block = False
				inner_product_block = False
				ReLU_block = False
				implementation_input_block = False
				norm_block = False
				dropout_block = False
				start_conv = line_number - 2

		elif 'type: "Pooling"' in line:
			if (bracket_count != 0):
				data_block = False
				convolution_block = False
				pooling_block = True
				inner_product_block = False
				ReLU_block = False
				implementation_input_block = False
				norm_block = False
				dropout_block = False
				start_pooling = line_number - 2

		elif 'type: "InnerProduct"' in line:
			if (bracket_count != 0):
				data_block = False
				convolution_block = False
				pooling_block = False
				inner_product_block = True
				ReLU_block = False
				implementation_input_block = False
				norm_block = False
				dropout_block = False
				start_product = line_number - 2

		elif 'type: "ReLU"' in line:
			if (bracket_count != 0):
				data_block = False
				convolution_block = False
				pooling_block = False
				inner_product_block = False
				implementation_input_block = False
				ReLU_block = True
				norm_block = False
				dropout_block = False

		elif 'type: "Input"' in line:
			if (bracket_count != 0):
				data_block = False
				convolution_block = False
				pooling_block = False
				inner_product_block = False
				ReLU_block = False
				finish_program_layer = False
				implementation_input_block = True
				norm_block = False
				dropout_block = False
				start_implementaion_input = line_number - 2

		elif 'type: "LRN"' in line:
			if (bracket_count != 0):
				data_block = False
				convolution_block = False
				pooling_block = False
				inner_product_block = False
				ReLU_block = False
				finish_program_layer = False
				implementation_input_block = False
				norm_block = True
				dropout_block = False
				start_norm = line_number - 2

		elif 'type: "Dropout"' in line:
			if (bracket_count != 0):
				data_block = False
				convolution_block = False
				pooling_block = False
				inner_product_block = False
				ReLU_block = False
				finish_program_layer = False
				implementation_input_block = False
				norm_block = False
				dropout_block = True
				start_dropout = line_number - 2

		if 'convolution_param' in line:
			conv_para_line = line_number

		if (bracket_count == 0):
			#on finishing line of a block
			end_line = line_number
			if (data_block == True):
				copyFromLinetoLine(start_data, end_line, directory, destination, 'a+')
				data_block = False
				num_data_layer += 1
				copyfile(destination, 'conv_temp.txt')
				with open('conv_temp.txt', 'r') as conv_temp, open(destination, 'w') as final_file:
					all_lines = conv_temp.readlines()
					for i in range(0, len(all_lines)):
						this_line = all_lines[i]
						if 'batch_size' in this_line:
							original_batch_size = this_line.split(' ')[5].lstrip().rstrip()
							batch_size = int(original_batch_size)
							batch_size = batch_size_processor(batch_size)
							final_file.write("    batch_size: " + str(batch_size) + '\n')
						else:
							final_file.write(this_line)

				# program layer handler
				if (num_data_layer == 2):
					# assume there are only 2 data layers
					# at this point data layer are setup, insert program layer and pad layer
					# assume input images have 3 channels, append Pad layer, otherwise use file setup_noPad.txt
					# both files set the first layer after program layers and pad layers with name "half1"
					copyFromLinetoLine(1, 38, program_pad, destination, 'a+')
					finish_program_layer = True

			# data input is for implementation, only one data block with type "Input" will be there
			elif (implementation_input_block == True and convolution_block == False and finish_program_layer == False and pooling_block == False and data_block == False and ReLU_block == False and inner_product_block == False and norm_block == False and dropout_block == False):
				copyFromLinetoLine(start_implementaion_input, end_line, directory, destination, 'a+')
				implementation_input_block = False
				copyFromLinetoLine(1, 38, program_pad, destination, 'a+')
				finish_program_layer = True

			# norm layer
			elif (norm_block ==True and implementation_input_block == False and convolution_block == False and pooling_block == False and data_block == False and ReLU_block == False and inner_product_block == False and dropout_block == False):
				copyFromLinetoLine(start_norm, end_line, directory, destination, 'a+')
				norm_block = False

			# dropout layer
			elif (dropout_block == True and norm_block == False and implementation_input_block == False and convolution_block == False and pooling_block == False and data_block == False and ReLU_block == False and inner_product_block == False):
				copyFromLinetoLine(start_dropout, end_line, directory, destination, 'a+')
				dropout_block = False

			# assume if current block is none of these, then reach the end of the layers, need to insert programming layers
			elif (inner_product_block == False and pooling_block == False and convolution_block == False and data_block == False and ReLU_block == False and finish_end_program == False and norm_block == False and dropout_block == False):
				copyFromLinetoLine(1, 18, end_program_layers, destination, 'a+')
				finish_end_program = True
				copyFromLinetoLine(current_block_start_line, len(raw_text), directory, destination, 'a+')
				done = True

			# convolution layer handler
			elif (convolution_block == True and pooling_block == False and data_block == False and ReLU_block == False and implementation_input_block == False and norm_block == False and dropout_block == False):
				# set the flag back
				convolution_block = False

				inner_bracket_count = -1			
				copyFromLinetoLine(start_conv, end_line, directory, destination, 'a+')
				copyfile(destination, 'conv_temp.txt')
				with open('conv_temp.txt', 'r') as conv_temp, open(destination, 'w') as final_file:
					all_lines = conv_temp.readlines()
					for i in range(0, len(all_lines)):
						this_line = all_lines[i]
						if 'layer {' in this_line:
							inner_bracket_count = 1
						elif '{' in this_line:
							inner_bracket_count += 1
						elif '}' in this_line:
							inner_bracket_count -= 1

						if 'type: "OCLCRHWCN"' in this_line:
							inner_conv_block = True
							inner_pooling_block = False
							inner_inner_product_block = False

						if 'type: "Convolution"' in this_line:
							inner_conv_block = True
							inner_pooling_block = False
							inner_inner_product_block = False

						if 'type: "OCLPoolingHWCN"' in this_line:
							inner_conv_block = False
							inner_pooling_block = True
							inner_inner_product_block = False

						if 'type: "Pooling"' in this_line:
							inner_conv_block = False
							inner_pooling_block = True
							inner_inner_product_block = False

						if this_line.strip() == 'type: "Convolution"':
							###### half1 name is not fixed
							if (inner_conv_block == True):
								final_file.write('  type: "OCLCRHWCN"\n')
						elif 'num_output' in this_line:
							if (inner_conv_block == True):
								num_output = this_line.split(' ')[5].lstrip().rstrip() #do not know why 5 gives the number
								temp_num = int(num_output)
								if (temp_num%16 != 0):
									temp_num = round_up(temp_num, 16)
								final_file.write('    num_output: ' + str(temp_num) + '\n')
						elif 'stride' in this_line:
							if (inner_conv_block == True):
								final_file.write('    stride: 1' + '\n')
							if (inner_pooling_block == True):
								original_stride_size = this_line.split(' ')[5].lstrip().rstrip()
								stride_size = int(original_stride_size)
								if (stride_size == 2) or (stride_size == 3):
									final_file.write(this_line)
								else:
									# assume set stride value for pooling layer to 3 if original stride is bigger then 3, or 2 if was 1
									if (stride_size > 3):
										final_file.write('    stride: 3' + '\n')
									else:
										final_file.write('    stride: 2' + '\n')
						elif 'convolution_param' in this_line:
							if (inner_conv_block == True):
								if 'engine: OCL' in all_lines[i+1]:
									final_file.write('  convolution_param {\n')
								else:
									final_file.write('  convolution_param {\n    engine: OCL\n    subengine: DIRECT\n')
						elif (inner_bracket_count == 0 and inner_conv_block == True):
							# reaches the end of this convolution block
							if 'relu' in all_lines[i-2]:
								inner_conv_block = False
								final_file.write(this_line)
								continue
							else:
								if 'type: "ReLU"' in raw_text[line_number+2].rstrip():
									final_file.write('  ocl_enable: true\n  cr_param {\n    relu: 1\n  }\n}\n')

								# handle conv - pooling - relu with max pooling case
								elif (pooling_relu_detector(directory, line_number) == True):
									final_file.write('  ocl_enable: true\n  cr_param {\n    relu: 1\n  }\n}\n')
								else:
									final_file.write('  ocl_enable: true\n  cr_param {\n    relu: 0\n  }\n}\n')
							inner_conv_block = False
						elif (1):
							final_file.write(this_line)

			# pooling layer handler
			elif (pooling_block == True and convolution_block == False and data_block == False and ReLU_block == False and implementation_input_block == False and norm_block == False and dropout_block == False):
				# set the flag back
				pooling_block = False

				inner_bracket_count = -1
				copyFromLinetoLine(start_pooling, end_line, directory, destination, 'a+')
				copyfile(destination, 'conv_temp.txt')
				with open('conv_temp.txt', 'r') as conv_temp, open(destination, 'w') as final_file:
					all_lines = conv_temp.readlines()
					for i in range(0, len(all_lines)):
						this_line = all_lines[i]
						if 'layer {' in this_line:
							inner_bracket_count = 1
						elif '{' in this_line:
							inner_bracket_count += 1
						elif '}' in this_line:
							inner_bracket_count -= 1

						if 'type: "OCLCRHWCN"' in this_line:
							inner_conv_block = True
							inner_pooling_block = False
							inner_inner_product_block = False

						if 'type: "Convolution"' in this_line:
							inner_conv_block = True
							inner_pooling_block = False
							inner_inner_product_block = False

						if 'type: "OCLPoolingHWCN"' in this_line:
							inner_conv_block = False
							inner_pooling_block = True
							inner_inner_product_block = False

						if 'type: "Pooling"' in this_line:
							inner_pooling_block = True
							inner_conv_block = False
							inner_inner_product_block = False

						if 'type: "Pooling"' in this_line:
							if (inner_pooling_block == True):
								final_file.write('  type: "OCLPoolingHWCN"\n')
						elif 'pool:' in this_line:
							if (inner_pooling_block == True):
								final_file.write('    pool: MAX\n')
						elif 'kernel_size:' in this_line:
							if (inner_pooling_block == True):
								original_kernel_size = this_line.split(' ')[5].lstrip().rstrip()
								kernel_size = int(original_kernel_size)
								if (kernel_size == 2) or (kernel_size == 3):
									final_file.write(this_line)
								else:
									# assume kernel size for pooling layer to 3 if original kernel size is bigger than 3, or 2 if was 1
									if (kernel_size > 3):
										final_file.write('    kernel_size: 3' + '\n')
									else:
										final_file.write('    kernel_size: 2' + '\n')
							elif (inner_conv_block == True):
								final_file.write(this_line)
						elif 'stride:' in this_line:
							if (inner_pooling_block == True):
								final_file.write('    stride: 2' + '\n')
							elif (inner_conv_block == True):
								final_file.write('    stride: 1' + '\n')
						elif (inner_bracket_count == 0):
							inner_pooling_block = False
							final_file.write(this_line)
						else:
							final_file.write(this_line)

			# inner product layer handler
			elif (inner_product_block == True and pooling_block == False and convolution_block == False and data_block == False and ReLU_block == False and implementation_input_block == False and norm_block == False and dropout_block == False):
				# set the flag back
				inner_product_block = False

				inner_bracket_count = -1
				copyFromLinetoLine(start_product, end_line, directory, destination, 'a+')
				copyfile(destination, 'conv_temp.txt')
				with open('conv_temp.txt', 'r') as conv_temp, open(destination, 'w') as final_file:
					all_lines = conv_temp.readlines()
					for i in range(0, len(all_lines)):
						this_line = all_lines[i]
						if 'layer {' in this_line:
							inner_bracket_count = 1
						elif '{' in this_line:
							inner_bracket_count += 1
						elif '}' in this_line:
							inner_bracket_count -= 1

						if 'type: "OCLCRHWCN"' in this_line:
							inner_conv_block = True
							inner_pooling_block = False
							inner_inner_product_block = False

						if 'type: "Convolution"' in this_line:
							inner_conv_block = True
							inner_pooling_block = False
							inner_inner_product_block = False

						if 'type: "OCLPoolingHWCN"' in this_line:
							inner_conv_block = False
							inner_pooling_block = True
							inner_inner_product_block = False

						if 'type: "Pooling"' in this_line:
							inner_pooling_block = True
							inner_conv_block = False
							inner_inner_product_block = False

						if 'type: "InnerProduct"' in this_line:
							inner_pooling_block = False
							inner_conv_block = False
							inner_inner_product_block = True

						if 'type: "OCLHWCNInnerProduct"' in this_line:
							inner_pooling_block = False
							inner_conv_block = False
							inner_inner_product_block = True

						if 'type: "InnerProduct"' in this_line:
							if (inner_inner_product_block == True):
								final_file.write('  type: "OCLHWCNInnerProduct"\n')
						elif 'num_output:' in this_line:
							if (inner_inner_product_block == True):
								product_output = this_line.split(' ')[5].lstrip().rstrip()
								product_num_output = int(product_output)
								last_inner_product_num_output = product_num_output
								if (product_num_output % 16 != 0):
									product_num_output = round_up(product_num_output, 16)
								final_file.write('    num_output: ' + str(product_num_output) + '\n')
							elif (inner_conv_block == True):
								final_file.write(this_line)
						elif (inner_bracket_count == 0 and inner_inner_product_block == True):
							# reaches the end of this convolution block
							if 'relu' in all_lines[i-2]:
								inner_conv_block = False
								final_file.write(this_line)
								continue
							else:
								if 'type: "ReLU"' in raw_text[line_number+2].rstrip():
									final_file.write('  cr_param {\n    relu: 1\n  }\n}\n')
								else:
									final_file.write('  cr_param {\n    relu: 0\n  }\n}\n')
							inner_inner_product_block = False
						else:
							final_file.write(this_line)

# top buttom name handler
# requiement: data layers must with name "data" and "label"

accuracy_layer = False
SoftmaxWithLoss_layer = False
accuracy_first_bottom = True
SoftmaxWithLoss_first_bottom = True

inner_product_count = 0
current_inner_product_count = 0

copyfile(destination, 'final_process.txt')
with open('final_process.txt', 'r') as conv_temp, open(destination, 'w') as final_file:
	all_lines = conv_temp.readlines()
	print 'Sorting top bottom names...'
	for i in range(0, len(all_lines)):
		this_line = all_lines[i]
		if 'layer {' in this_line:
			accuracy_layer = False
			SoftmaxWithLoss_layer = False
			final_file.write(this_line)
		elif 'type: "OCLHWCNInnerProduct"' in this_line:
			inner_product_count += 1
			final_file.write(this_line)

		# handle Accuracy layer seperatly because it has two bottom lauers with different naming style
		elif 'type: "Accuracy"' in this_line:
			accuracy_layer = True
			final_file.write(this_line)

		# handle softmax with loss seperatly because it has two bottom lauers with different naming style
		elif 'type: "SoftmaxWithLoss"' in this_line:
			SoftmaxWithLoss_layer = True
			final_file.write(this_line)

		elif 'top: ' in this_line:
			name = this_line.split(' ')[3].lstrip().rstrip().split('"')[1]
			if (name != 'label' and accuracy_layer == False):
				next_bottom = this_line.split(' ')[3].lstrip().rstrip().split('"')[1]
			final_file.write(this_line)
		elif 'bottom: ' in this_line:
			if (accuracy_layer == True):
				if (accuracy_first_bottom == True):
					final_file.write('  bottom: "' + next_bottom + '"\n')
					accuracy_first_bottom = False
				else:
					final_file.write('  bottom: "label"\n')
			elif (SoftmaxWithLoss_layer == True):
				if (SoftmaxWithLoss_first_bottom == True):
					final_file.write('  bottom: "' + next_bottom + '"\n')
					SoftmaxWithLoss_first_bottom = False
				else:
					final_file.write('  bottom: "label"\n')
			else:
				final_file.write('  bottom: "' + next_bottom + '"\n')
		else:
			final_file.write(this_line)

# last inner product num_output handler

change_next_line_num_output = False
copyfile(destination, 'final_process.txt')
with open('final_process.txt', 'r') as conv_temp, open(destination, 'w') as final_file:
	print 'Finalizing inner product layer parameters...'
	all_lines = conv_temp.readlines()
	for i in range(0, len(all_lines)):
		this_line = all_lines[i]
		if 'type: "OCLHWCNInnerProduct"' in this_line:
			current_inner_product_count += 1
			final_file.write(this_line)
		elif 'inner_product_param' in this_line:
			if (current_inner_product_count == inner_product_count):
				change_next_line_num_output = True
			final_file.write(this_line)
		elif 'num_output' in this_line:
			if (change_next_line_num_output == True):
				final_file.write('    num_output: ' + str(last_inner_product_num_output) + '\n')
				change_next_line_num_output = False
			else:
				final_file.write(this_line)
		else:
			final_file.write(this_line)

copyfile(destination, 'FPGA_Caffe.prototxt')
print 'Cleaning...'
os.remove("FPGA.txt")
os.remove("conv_temp.txt")
os.remove("final_process.txt")
os.remove("temp.txt")
print 'Done!'
print 'Check output file in ' + output_file