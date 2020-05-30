import numpy as np

def maxpooling(image, f=2 ,s=2):
	'''
	A function that implements maxpooling downsizing on an image array of shape (number_channels, height of image, width of image).

	f denotes the kernel size
	s denotes the stride value
	'''

	number_channels, h_prev, w_prev = image.shape

	#calculate output dimensions
	h = int((h_prev - f)/s)+1
	w = int((w_prev - f)/s)+1


	#create output matrix
	downsampled = np.zeros((number_channels, h, w))

	#slide over window using stride s and take the maximum value
	for i in range(number_channels):
		current_y = output_y = 0
		while current_y + f <= h_prev:
			current_x = output_x = 0
			while current_x +f < w_prev:
				downsampled[i, output_y, output_x] = np.max(image[i, current_y:current_y + f, current_x:current_x+f])
				current_x += s
				output_x += 1
			current_y += s
			output_y += 1
	return downsampled



def nanargmax(arr):
    '''
    returns the largest non-nan value in an array.
    Output is a ordered pair tuble using unravel_index
    '''
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs



def maxpoolBackward(dpool, orig, f, s):
	'''
	Backpropogation through a maxpooling layer.
	Gradients are passed through the indices of greatest value in the original maxpooling during the forward step.


	Note:There is no gradient with respect to non maximum values, since changing them slightly does not affect the output.
	Further the max is locally linear with slope 1, with respect to the input that actually achieves the max. Thus, the gradient 
	from the next layer is passed back to only that neuron which achieved the max. All other neurons get zero gradient.
	'''
	(n_c, orig_dim, _) = orig.shape
	d_out = np.zeros(orig.shape)

	for curr_c in range(n_c):
		curr_y = out_y = 0 
		while curr_y + f <= orig_dim:
			curr_x = out_x = 0
			while curr_x + f <=orig_dim:
			# find largest value in input for curerent window	
			(a, b) = nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
			d_out[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]

			curr_x += s
			out_x+=1
		curr_y += s
		out_y += 1
	return d_out
