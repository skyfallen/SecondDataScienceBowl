from __future__ import print_function

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K 

# from original vgg 
# from keras.optimizers import SGD
# import cv2, numpy as np

LR = 0.0001
REG = 0.0001

def get_name():
    return "LR" + str(LR) + "_REG" + str(REG)

def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)

#def VGG_16(weights_path=None):
def get_model(img_size):

    img_size2 = img_size + img_size / 2
    img_size3 = img_size * 2
    img_size4 = img_size * 2 + img_size / 2

    model = Sequential()	
    model.add(Activation(activation=center_normalize, input_shape=(30, img_size, img_size)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size2, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size3, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(img_size4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, W_regularizer=l2(REG)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1024, W_regularizer=l2(1e-03)))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='rmse')
    return model

