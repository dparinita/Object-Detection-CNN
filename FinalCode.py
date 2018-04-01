#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:35:10 2016

@author: computational
"""

import FinalFlow as ff
from PIL import Image
import os, glob

pathToVideo="/home/computational/Downloads/FinalFlow/InputVideo/FinalSample1.mp4" 
pathToFrames="/home/computational/Downloads/FinalFlow/OutputFrames/image%03d.jpg"
folderToFrames="/home/computational/Downloads/FinalFlow/OutputFrames"
weightFile="/home/computational/SpyderCodes/testweights.h5"
framesToPick="1"
everySecond="5"

try:    
    ff.extractFramesFromVideo(pathToVideo, pathToFrames, framesToPick, everySecond)
    print "Extracted frames"
except:
    print "Couldn't extract frames"
  
try:
    os.chdir(folderToFrames) ##to set
    for file in glob.glob("*.jpg"):
        print(file)
        pathToFile = folderToFrames + "/" + file
        image = Image.open(pathToFile)
        #image.show()
        result = ff.runModel(pathToFile, weightFile)
        print result
except:
    print "Error in resizing"
