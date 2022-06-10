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

import matplotlib.pyplot as plt
plt.switch_backend('agg')


PHOTOSIZE = 32
RANSTATE = 1000

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
