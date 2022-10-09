import timeit
from keras.model import Model, Sequential
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout, Flatten, Conv2D, MaxPool12D, Activation, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLRonPlateau
from keras import backend as K
from keras.utils.generic_utils import get_costum_objects
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import mnist
import tensorflow as tf
from IPython.display import Image

import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

numpy.random.seed(7)

def plot_(history):
    training_loss1 = history.history['loss']
    test_loss1 = history.history['val_loss']
    epoch_count = range(1, len(training_loss1) + 1)
    plt.plot(epoch_count, training_loss1, 'r--')
    plt.plot(epoch_count, test_loss1, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();

def x1(x):
    return (K.pow(x,1))
get_custom_objects().update({"x1": Activation(x1)})

def x2(x):
    return(K.pow(x,2))/2
get_custom_objects().update({"x2": Activation(x2)})

def x3(x):
    return(K.pow(x,3))/6
get_custom_objects().update({"x3": Activation(x3)})

def x4(x):
    return(K.pow(x,4))/24
get_custom_objects().update({"x4": Activation(x4)})

def x5(x):
    return(K.pow(x,5))/120
get_custom_objects().update({"x5": Activation(x5)})

def x6(x):
    return(K.pow(x,6))/720
get_custom_objects().update({"x6": Activation(x6)})

def x7(x):
    return(K.pow(x,7))/5040
get_custom_objects().update({"x7": Activation(x7)})

def x8(x):
    return(K.pow(x,8))/40320
get_custom_objects().update({"x8": Activation(x8)})

