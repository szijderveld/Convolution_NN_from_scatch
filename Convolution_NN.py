'''
The network created takes a training image of 28*28 pixels and outputs a class labal (10 in total). The architecture used to do this uses two consective 
convolution layers followed by a max pooling operation to extract features from the input image. After this the represtation is flattened an passed through a 
Multi-Layer Perceptron to carry out the final classification task.


Layer ordering:


1. Input 28*28 image
2. Strided convolution with eight 5*5 fiters and a stride of 1
3. Strided convolution with eight 5*5 filters and a stride of 1
4. Max pooling operation with a 2*2 filter and a stride of 2
5. Flatten previous layer into vector
6. Dense layer with 128 neurons
7 10-outit output layer (classification layer)
'''

import numpy as np 
def extract_images(filename, num_samples, image_width=28):
    '''
    Extracts images files and reshapes values into matrix (num_images, image_width**2)
    '''
    print('Extracting', filename)
    with gzip.open(filename,'r') as f:
        f.read(16)
        buf = f.read(image_width * image_width * num_samples)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #print(data.shape)
        data = data.reshape(num_samples, image_width*image_width)
    return data

def extract_labels(filename, num_samples):
    data = []
    with gzip.open(filename,'r') as f:
        f.read(8)
        for i in range(0,num_samples):
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            data.append(labels)
    return data


def initialiseFilter(size, scale=1.0):
    '''
    Initialise filter using normal distribution with standard deviation proportional to the square root of the number of units
    '''
    stddev = scale / np.sqrt((size))
    return  np.random.normal(loc=0, scale=stddev, size=size)
def initialiseWeights(size):
    '''
    Initialise weights with random normal distribution
    '''
    return np.random.standard_normal(size=size)*0.01


from Softmax import softmax
from convolution import convolution
def conv(image, label, params, conv_s, pool_f, pool_s):
    '''
    Combine the forward and backward propogation to build a method that takes the input parameters and hyperparamets as inputs and outputs gradient and loss
    '''

    [f1, f2, w3, w4, b1, b2, b3, b4] = params   #filters, weights and biases

    #############################################
    ###########Forward operation#################
    #############################################

    conv1 = convolution(image, f1, b1, conv_s) 
    conv1[conv1<=0] = 0 #apply ReLU non-linearity 

    cov2 = convolution(conv1, f2, b2, conv_s)
    conv2[conv2<=0] = 0

    pooled = maxpool(conv2, pool_f, pool_s) #maxpooling

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2*dim2*dim2, 1))    #flatten pooled layer

    z = w3.dot(fc) + b3 #pass flattened pool through first fully connected layer
    z[z<=0] = 0 #pass through ReLU function

    out = w4.dot(z) + b4 #pass through second layer

    probs = softmax(out)  #apply softmax activation function to find prodicted probabilities



    #############################################
    ############# Loss ##########################
    #############################################
    loss = categoricalCrossEntropy(probs, label) 



    ##############################################
    ############ Backward operation ##############
    ##############################################

    d_out = probs - label      #derivate of loss w.r.t final dense layer

    dw4 = d_out.dot(z.T)                               #loss gradient weights
    db4 = np.sum(d_out , axis=1).reshape(b4.shape)     #loss gradient of biases


    dz = w4.T.dot(d_out)        #loss gradient of first dense layer outputs
    dz[z<=0] = 0                #ReLU
    dw3 = dz.dot(fc.T)          #loss function os weights
    db3 = np.sum(dz, axis=1).reshape(b3.shape)


    dfc = w3.T.dot(dz)    #loss gradient of fully connested pooling layer
    dpool = dfc.reshape(pooled.shape)       #reshape into into dimension of pooling layer

    dconv2 = maxoolBackward(dpool, conv2, pool_f, pool_s)   
    dconv2[conv2<=0] = 0


    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_s)
    dconv1[conv1<=0] = 0

    dimage, df1, db1 = convolutionBackward(dconv1, image, f1, convs)


    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]


    return grads, loss




def adamGB(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    update parameter through Adams gradient descent
    '''

    [f1, f2, w3, w4, b1, b2, b3, b4] = params


    X = batch[:,0]  #batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1]  #batch labels

    cost_= 0
    batch_size = len(batch)

    #initialize gradient and momentum, RMS params
    
    