'''
In order to measure acuracy of our model we have to make use of a loss function. A loss function when predicting multiple output clasees is the categorical 
cross entropy loss function:

H(y,y_bar) = sum(y_i*log(1/y_bar_i)) = -sum(y_ilog(y_bar_i))

'''



def categoricalCrossEntropy(probs, label):

	return =np.sum(label*np.log(probs))