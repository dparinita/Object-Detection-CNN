#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:41:47 2016

@author: computational
"""
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from PIL import Image
import cv2, numpy as np
import os,glob

pathToVideo="/home/computational/Downloads/FinalFlow/InputVideo/FinalSample.mp4" 
pathToFrames="/home/computational/Downloads/FinalFlow/OutputFrames/image%03d.jpg"
folderToFrames="/home/computational/Downloads/FinalFlow/OutputFrames"
weightFile="/home/computational/SpyderCodes/testweights.h5"
framesToPick="1"
everySecond="5"
num_classes=10

def Model(weightFile):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, 32, 32)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))


    if weightFile:
        model.load_weights(weightFile)

    return model
    
def runModel(file, weightFile):
    im = cv2.resize(cv2.imread(file), (32, 32)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = Model(weightFile)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    ##print np.argmax(out)
    output = np.argmax(out)
    ##print output
    ##line = linecache.getline("/home/computational/SpyderCodes/synset_words.txt", output+1)
    return output

def extractFramesFromVideo(pathToVideo, pathToFrames, framesToPick, everySecond):
    string = "ffmpeg -i " + pathToVideo + " -r " + framesToPick + "/" + everySecond + " " + pathToFrames
    os.system(string)
