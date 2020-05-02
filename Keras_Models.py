### Base Convolutional Network Models
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten, Dropout
from keras.layers import Activation
from keras.utils import plot_model

import cv2

def generate_simple_model(shape): # Best is 0.71
  conv_net = Sequential()
  #Using a sequential model makes it easy to build custom nets

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=shape))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Flatten())
  
  conv_net.add(Dense(64, activation="relu"))
  conv_net.add(Dropout(0.5))
  conv_net.add(Dense(5, activation="softmax"))

  plot_model(conv_net)

  conv_net.summary()
  return conv_net


def generate_simple_batch_model(shape): # Best is 0.71
  conv_net = Sequential()
  #Using a sequential model makes it easy to build custom nets

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=shape))
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3)))
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Flatten())
  
  conv_net.add(Dense(64, activation="relu"))
  conv_net.add(Dropout(0.5))
  conv_net.add(Dense(5, activation="softmax"))

  plot_model(conv_net)

  conv_net.summary()
  return conv_net

def generate_base_model(shape): # Best is 0.71
  conv_net = Sequential()
  #Using a sequential model makes it easy to build custom nets

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=shape))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Flatten())
  
  conv_net.add(Dense(64, activation="relu"))
  conv_net.add(Dropout(0.5))
  conv_net.add(Dense(5, activation="softmax"))

  plot_model(conv_net)

  conv_net.summary()
  return conv_net

def generate_batch_norm_model(shape): # Best is 0.71
  conv_net = Sequential()
  #Using a sequential model makes it easy to build custom nets

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=shape))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Flatten())
  
  conv_net.add(Dense(64, activation="relu"))
  conv_net.add(Dropout(0.5))
  conv_net.add(Dense(5, activation="softmax"))

  plot_model(conv_net)

  conv_net.summary()
  return conv_net


def generate_deep_batch_norm_model(shape): # Best is 0.71
  conv_net = Sequential()
  #Using a sequential model makes it easy to build custom nets

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=shape))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=32, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))

  conv_net.add(Conv2D(filters=64, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))
  
  conv_net.add(Conv2D(filters=128, kernel_size=(3,3)))
  conv_net.add(BatchNormalization())
  conv_net.add(Activation('relu'))
  conv_net.add(MaxPool2D(pool_size=(2,2)))


  conv_net.add(Flatten())
  
  conv_net.add(Dense(64, activation="relu"))
  conv_net.add(Dropout(0.5))
  conv_net.add(Dense(5, activation="softmax"))

  plot_model(conv_net)

  conv_net.summary()
  return conv_net

def get_model_list():
  model_list = [
    #(generate_simple_model, '3-layer-conv-model'),
    #(generate_batch_norm_model, 'batch-conv-model'),
    (generate_simple_batch_model, '3-layer-batch-model'),
    (generate_base_model, 'conv-model'),
    #(generate_deep_batch_norm_model, 'deep-batch-model')
  ]
  return model_list