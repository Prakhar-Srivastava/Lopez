import os
import cv2
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def imgPath(sketchPath):
    filename = sketchPath.split('/')[-1].lower()
    dirs = sketchPath[:sketchPath.rfind('/')]
    dirs = dirs[:dirs.rfind('/')]
    name = filename[0] + filename[filename.find('-'):filename.rfind('-')] +\
    '.jpg'
    return dirs + '/photos/' + name


rgb2gray = lambda x: np.dot(x[..., :3], [.2989, .5870, .1140])
dir = './sketches/'
for file in os.listdir(dir):
    path = dir+file
    sketch = np.invert(plt.imread(path))
    img = rgb2gray(plt.imread(imgPath(path)))
    print('shape of sketch:', sketch.shape, 'shape of image:', img.shape)
