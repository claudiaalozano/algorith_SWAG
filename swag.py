from ast import increment_lineno
import tensorflow
import timeit
from keras.model import Model,Sequential
from keras.layers import Input, Embedding, LSTM, LSTM, Dense, concatenate, Dropout, Flatten, Conv2D, MaxPool12D, Activation, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLRonPlateau
from keras import backend as k
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
%matplotlib inline
from sklearn.moder_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

