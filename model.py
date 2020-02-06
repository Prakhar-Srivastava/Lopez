import pandas as pd
import numpy as np
from joblib import load as jLoad, dump
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D,\
                Conv2DTranspose, UpSampling2D, LeakyReLU, Dropout,\
                BatchNormalization, SeparableConv2D
from keras.optimizers import Adam, RMSprop

class Model:
    def __init__(self, img_rows, img_cols, channel=1, discriminator=None,\
                generator=None, adversarialModel=None,\
                discriminatorModel=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.__discriminator__ = discriminator
        self.__generator__ = generator
        self.__am__ = adversarialModel
        self.__dm__ = discriminatorModel


    def discriminator(self):
        if not self.__discriminator__:
            dropout = 0.2
            convOut = 128
            input_shape = (self.img_rows, self.img_cols, self.channel)
            self.__discriminator__ = Sequential()
            self.__discriminator__.add(Conv2D(convOut, 5, strides=(1, 1),\
                input_shape=input_shape, padding='same'))
            self.__discriminator__.add(LeakyReLU(alpha=0.12))
            self.__discriminator__.add(Dropout(dropout))

            self.__discriminator__.add(Conv2D(convOut * 2, 5, strides=(1, 1),\
                padding='valid'))
            self.__discriminator__.add(LeakyReLU(alpha=0.2))
            self.__discriminator__.add(Dropout(dropout * 1.4))

            self.__discriminator__.add(SeparableConv2D(convOut * 4, 5,\
                strides=(1, 1)))
            self.__discriminator__.add(LeakyReLU(alpha=0.2))
            self.__discriminator__.add(Dropout(dropout * 1.6))

            self.__discriminator__.add(Conv2D(convOut * 8, 5, strides=(1, 1),\
                padding='valid'))
            self.__discriminator__.add(LeakyReLU(alpha=0.15))
            self.__discriminator__.add(Dropout(dropout / 1.3))

            self.__discriminator__.add(Flatten())
            self.__discriminator__.add(Dense(10))
            self.__discriminator__.add(Activation('relu'))
            self.__discriminator__.add(Dense(1))
            self.__discriminator__.add(Activation('sigmoid'))
            self.__discriminator__.summary()
        return self.__discriminator__


    def generator(self):
        if not self.__generator__:
            dropout = 0.2
            depth = 4 * 128
            self.__generator__ = Sequential()
            self.__generator__.add(Dense(62*50*depth, input_dim=100))
            self.__generator__.add(BatchNormalization(momentum=0.9))
            self.__generator__.add(Activation('relu'))
            self.__generator__.add(Reshape((62, 50, depth)))
            self.__generator__.add(Dropout(dropout))

            self.__generator__.add(UpSampling2D())
            self.__generator__.add(Conv2DTranspose(int(depth/2), 5,\
                padding='same'))
            self.__generator__.add(BatchNormalization(momentum=0.9))
            self.__generator__.add(Activation('relu'))

            self.__generator__.add(UpSampling2D())
            self.__generator__.add(Conv2DTranspose(int(depth/4), 5,\
                padding='same'))
            self.__generator__.add(BatchNormalization(momentum=0.9))
            self.__generator__.add(Activation('relu'))

            self.__generator__.add(Conv2DTranspose(int(depth/8), 5,\
                padding='same'))
            self.__generator__.add(BatchNormalization(momentum=0.9))
            self.__generator__.add(Activation('relu'))

            self.__generator__.add(Conv2DTranspose(1, 5, padding='same'))
            self.__generator__.add(Activation('sigmoid'))
            self.__generator__.summary()
        return self.__generator__


    def adversarialModel(self):
        if not self.__am__:
            optimizer = RMSprop(lr=1e-5, decay=3e-8)
            self.__am__ = Sequential()
            self.__am__.add(self.generator())
            self.__am__.add(self.discriminator())
            self.__am__.compile(optimizer=optimizer, loss='binary_crossentropy',\
                metrics=['accuracy'])
        return self.__am__


    def discriminatorModel(self):
        if not self.__dm__:
            optimizer = RMSprop(lr=0.0002, decay=6e-8)
            self.__dm__ = Sequential()
            self.__dm__.add(self.discriminator())
            self.__dm__.compile(loss='binary_crossentropy', optimizer=optimizer,\
                metrics=['accuracy'])
        return self.__dm__


def save(path, model):
    print('Saving to %s...'%path)
    dump(model, path)


def load(path=None, img_rows=248, img_cols=200):
    if not path:
        return Model(img_rows, img_cols)
    print('Loading model from %s...'%path)
    return jLoad(path)


def train(X, y, trainSteps=10000, batchSize=10):
    X_train = X.values.reshape((-1, 250, 200))[:, :248,:]
    y_train = y.values.reshape((-1, 250, 200))[:,:248, :]
    model = load()
    am = model.adversarialModel()
    dm = model.discriminatorModel()
    gen = model.generator()
    for i in range(trainSteps):
        images_train = X_train[np.random.randint(0, X_train.shape[0], size=batchSize), :, :].reshape((batchSize, 248, 200, -1))
        print('shape of images_train', images_train.shape)
        noise = np.random.uniform(-1.0, 1.0, size=[batchSize, 100])
        print('shape of noise', noise.shape)
        images_fake = gen.predict(noise)# .reshape((batchSize, 248, 200))
        print('shape of images_fake', images_fake.shape)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*batchSize, 1])
        y[batchSize:, :] = 0
        d_loss = dm.train_on_batch(x, y)

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batchSize, 100])
        a_loss = am.train_on_batch(noise, y)
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)



def test(dataset):
    print('Testing dataset')


# 1.6e6/4875
'''
62.5 * 50 * 128 * 4 = 1.6e6 = 62 * 50 * 4 * 128 + .5 * 50 * 4 * 128
==> 62.5 * 50 * 128 * 4 = 1.6e6 = 62 * 50 * 4 * 129.0322580645
62 * 50 * 4 * 129.03225806451611609950000000000004999...
'''
