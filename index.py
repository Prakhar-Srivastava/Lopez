import sys
import pandas as pd
# from preprocess import ppSketch, ppPhoto
from model import save, load, train, test, Model

action = sys.argv[0]
actions = {
    "save": lambda: save(sys.argv[2]),
    "load": lambda: load(sys.argv[2]),
    "train": lambda: train(pd.read_csv('./X_train.csv', header=None, skiprows=1),\
                            pd.read_csv('./y_train.csv', header=None, skiprows=1)),
    "test": lambda: test(pd.read_csv(sys.argv[2])),
}
actions[sys.argv[1]]()
