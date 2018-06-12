import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils


# It is good to know little basic of tensorflow since keras uses tensorflow backend 
# to develop the layr and computation graphs which is later used for Algorithmiuc differentiation for Gradient desecnt

# In this we are developing simple Sequential ANN 
'''

Structure of Neural Network

	Output layer   | Error reduction using Stocastic Gradient Descent method

	Hidden layer  
	
	Input layer

'''
# ANN hyper-parameters parameters
NB_CLASSES = 10
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
N_HIDDEN = 128
OPTIMIZER = 'SGD'


# in my implementation mnist foldr contains the data for training and testing
# this data can also be dowloaded using library mnist

# n x 28 x 28 pixel training data
X_train = np.load("mnist/x_train.npy")

# n x 1 training labels
Y_train = np.load("mnist/y_train.npy")

# n x 28 x 28 pixel testing data
X_test = np.load("mnist/x_test.npy")

# n x 1 testing labels
Y_test = np.load("mnist/y_test.npy")

# -------------------------------------------------------

# displaying the data set
#print(X_train[0])
# reshape for the neural network 28 x 28 = 784

# this will print the shape of input image 
print(X_train.shape[1], 'input pixel values per train samples')

# to use this as an input in neural network we need to make it into a vector
RESHAPED= 784

# Reshaping the data
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
# specify type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# normalization the data
# Normalizing makes it easier to optimize using Gradient descent
X_train /= 255
X_test /= 255




# data output: number of samples
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# data output: number of values / sample
print(X_train.shape[1], 'input pixel values per train samples')
print(X_test.shape[1], 'input pixel values per test samples')

# printing sample label
print(Y_train[0], 'label samples')

# convert label vectors to binary matrices of classes
# why do we convert ?
# Refer to one hot encoding !!! We do not want computer to take any value as higher or lower


Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)



# Alright you are now ready to prepare you first Sequential Neural Network using keras
# simple ANN model with just one layer
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

# compilation -  to get our parameters refined 
# compiling method
# compile(self, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# fit the model
# verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE)

# evaluation of the model
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])
