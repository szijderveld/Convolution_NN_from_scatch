def softmax(raw_predictions):
	'''
	passes raw predictions through softmax function, doing this will map all final dense alyer outputs to a vector whoes elements sum to one.
	'''

	out = np.ext(raw_predictions)
	return out/np.sum(out)