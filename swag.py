from tabnanny import verbose
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


batch_size = 128
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train+10
x_test = x_test+10

x_train /= 300
x_test /= 300
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

number_train = 60000
number_test = 10000

x_train = x_train[0:number_train, :]
x_test = x_test[0:number_test, :]

y_train = y_train[0:number_train]
y_test = y_test[0:number_test]


num_clases= 10
model = Sequential() # esto es un modelo secuencial
model.add(Dense(1024, activation='relu', input_shape=(784,))) # esto es una capa densa que sirve para conectar las neuronas
model.add(Dense(1024, activation='relu')) 
model.add(Dense(num_classes, activation='softmax')) # esto es la capa de salida que tiene 10 neuronas,lo que hemos establecido en num_classes

model.summary() # esto es para ver el resumen del modelo que hemos creado
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # esto es el optimizador que usaremos para entrenar el modelo

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy']) # esto es para compilar el modelo 
start = timeit.default_timer() # esto es para medir el tiempo de entrenamiento
history = model.fit(x_train, y_train, batch_size=batch_size, epochs= epochs, verbose=1, validation_data=(x_test, y_test)) # esto es para entrenar el modelo
end= timeit.default_timer() 
print(end-start) # con esto vemos el tiempo de entrenamiento

score= model.evaluate(x_test, y_test, verbose=0) # esto es para evaluar el modelo con los datos de test
print('Test loss:', score[0]) # esto es para ver el error de test
print('Test accuracy:', score[1]) # esto es para ver la precisión de test
plot_(history)

hidden_dimention = 500
model = Sequential()
in_ = Input(shape=(784,))

Layer_1_Act_X_1 = Dense(hidden_dimention, activation='X_1')(in_)
Layer_1_Act_X_2 = Dense(hidden_dimention, activation='X_2')(in_)
Layer_1_Act_X_3 = Dense(hidden_dimention, activation='X_3')(in_)

Concatenate_All_Layer = concatenate([in_,Layer_1_Act_X_1, Layer_1_Act_X_2, Layer_1_Act_X_3])

Out_Put = Dense(10, activation='linear')(Concatenate_All_Layer)
model = Model(in_, Out_Put)
model.summary()

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
start = timeit.default_timer()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
end = timeit.default_timer()
print(end-start)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
plot_(history)

hidden_dimention=500
model = Sequential()

in_ = Input(shape=(784,))
Layer_1_Act_X_1=Dense(hidden_dimention, activation='X_1')(in_)
################################################################
Layer_1_Act_X_2=Dense(hidden_dimention, activation='X_2')(in_)
###############################################################
Layer_1_Act_X_3=Dense(hidden_dimention, activation='X_3')(in_)
###############################################################
Layer_1_Act_X_4=Dense(hidden_dimention, activation='X_4')(in_)
###############################################################
Layer_1_Act_X_5=Dense(hidden_dimention, activation='X_5')(in_)
###############################################################
Layer_1_Act_X_6=Dense(hidden_dimention, activation='X_6')(in_)
###############################################################
Layer_1_Act_X_7=Dense(hidden_dimention, activation='X_7')(in_)
###############################################################
Layer_1_Act_X_8=Dense(hidden_dimention, activation='X_8')(in_)
###############################################################
Concatenate_First_Layer = concatenate([Layer_1_Act_X_1,Layer_1_Act_X_2,Layer_1_Act_X_3,Layer_1_Act_X_4,Layer_1_Act_X_5,Layer_1_Act_X_6,Layer_1_Act_X_7,Layer_1_Act_X_8])

Out_put=Dense(10, activation='linear')(Concatenate_First_Layer)
model = Model(in_ , Out_put)
model.summary()

optimizer =Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
Image(filename='model_2_layers.png')

start = timeit.default_timer()
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
end = timeit.default_timer()
print(end-start)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plot_(history)

