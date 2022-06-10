#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np

# from IPython.core.debugger import Tracer

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, UpSampling2D
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from trans_keras import transformer_generator, transformer_discriminator

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY


# In[2]:


PHOTOSIZE = 32
RANSTATE = 1000


# In[3]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        )
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = test_datagen.flow_from_directory(
        'ttsrb\\train',
        target_size=(PHOTOSIZE, PHOTOSIZE),
        batch_size=1,
        shuffle=False,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'watch',
        target_size=(PHOTOSIZE, PHOTOSIZE),
        batch_size=1,
        shuffle=False,
        class_mode='binary')

trainX=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
trainy=np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])
# trainy=to_categorical(np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())]))

testX=np.concatenate([validation_generator.next()[0] for i in range(validation_generator.__len__())])
testy=np.concatenate([validation_generator.next()[1] for i in range(validation_generator.__len__())])
# testy=to_categorical(np.concatenate([validation_generator.next()[1] for i in range(validation_generator.__len__())]))


# In[4]:


X_train = testX
# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=28, height=28, channels=3):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def __generator(self):
        """ Declare generator """

	model = transformer_generator()
	model.summary()

        return model

    def __discriminator(self):
        """ Declare discriminator """

        model = transformer_discriminator()
        model.summary()

        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, epochs=15000, batch = 32, save_interval = 300):

        for cnt in range(epochs):

            ## train discriminator
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
            legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)


            # train generator

            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)


    def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/ttsrb_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width, self.channels])
            plt.imshow( (image*255 ).astype(np.uint8) )
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    # (X_train, _), (_, _) = mnist.load_data()
    X_train = testX
    # Rescale -1 to 1
    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.astype(np.float32) / 255.0
    # X_train = np.expand_dims(X_train, axis=3)
    # print(X_train.shape)

    gan = GAN()
    gan.train(X_train)

# In[ ]:



if __name__ == '__main__':
    X_train = np.array([ testX[i] for i in range(len(testy)) if int(testy[i]) == 1 ])
    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 0.5) * 2
    # X_train = X_train.astype(np.float32) / 255.0
    # X_train = np.expand_dims(X_train, axis=3)
    print(X_train.shape)

    gan = GAN()
    gan.train(X_train)


# In[ ]:



