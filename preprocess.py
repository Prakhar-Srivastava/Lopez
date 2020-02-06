import os
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



def prepareTrain(dir):
    X = np.empty((50000,)).astype('int')
    y = np.empty((50000,)).astype('int')
    for file in os.listdir(dir):
        path = dir+file
        sketch = np.invert(plt.imread(path)).astype('int')
        X = np.vstack((X, sketch.flatten()))
        img = rgb2gray(plt.imread(imgPath(path))).astype('int')
        y = np.vstack((y, img.flatten()))
    pd.DataFrame(X).astype('int').to_csv('./X.csv', header=None, index=None)
    pd.DataFrame(y).astype('int').to_csv('./y.csv', header=None, index=None)

def ppSketch(sketch):
    print('ppSketch')


def ppPhoto(photo):
    print('ppPhoto')


prepareTrain('./sketches/')
