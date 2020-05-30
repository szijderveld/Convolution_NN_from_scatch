

import numpy as np 

def convolution(image, filt, bias, s=1):
	'''
	The concolution of two 3D arrays are found. 
	Inputs:
	- Image, a 3D array in format (number_chanels(e.g.RGB), image_dimension, image_dimension)
	- filt, filter array (n_filters, number_chanels, filter_dimension, filter_dimension)
	- bias of node
	- s = strike value
	'''
	(n_f, n_c_f, filt_dim, _)= filt.shape
	n_c , in_dim, _ = image.shape

	out_dim = int((in_dim-f)/s)+1			#calculate out dimension using standard formula

	assert n_c == n_c_f, "Number of channels of filter and input and equal"

	out = np.zeros((n_f, out_dim, out_dim)) 	#setup output concolution

	for current_filter in range(n_f):

		current_y = output_y = 0 
		#move actross filter vertically
		while current_y + filt_dim <= in_dim:
			current_x = output_x = 0

			#move horizontally
			while current_x + f <= in_dim:
				#convolution
				out[current_filter, output_y, output_x] = np.sum(filt[current_filter] * image[:, current_y:current_y+f, current_x: current_x+f])
				current_x += s
				output_x += 1

			current_y += s
			output_y += 1
	return output



def convolutionBackward(dconv_prev, conv_in, filt, s):
	'''
	Backward propogation through a convolution layer
	'''
	(number_filter, number_chanels, f, _) = filt.shape
	(_, orig_dim, _) = conv_in.shape

	###initialize derivatives
	d_out = np.zeros(conv_in.shape)
	d_filt = np.zeros(filt.shape)
	d_bias = np.zeros((number_filter,1))

	for current_filter in range(number_filter):
		current_y = output_y = 0
		while current_y + f <= orig_dim:
			current_x = output_x = 0
			while current_x + f <=orig_dim:
				#loss gradient of filter (used to update filter)
				d_filt[current_filter] += dconv_prev[current_filter, output_y, output_x] * conv_in[:, current_y:current_y + f, current_x: current_x + f]
				#loss gradient of input to the convolution operation (conv1 in the case of this network)
				d_out[:,current_y:current_y+f, current_x:current_x+f] +=  dconv_prev[current_filter, output_y, output_x] * filt[current_filter]
				current_x += s
				output_x += 1
			current_y += s
			output_y += 1
		#loss gradient of the bias
		d_bias[current_filter] = np.sum(dconv_prev[current_filter]) 
		return d_out, d_filt, d_bias





