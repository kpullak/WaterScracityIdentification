#!/bin/env python3

import keras
import csv
import pandas as pd
from pprint import pprint
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from shutil import copy2 as copy_file

im_path = "./TrainData/"
aug_data_path = './aug_training/'
val_data_path = './val_data/'
batch_size = 32 # 32
num_images_wanted = 8192 # for each label # 8192
num_validation_images = 100

def csv_to_list(filepath):
  df = pd.read_csv(filepath)
  return df.values.tolist()


if __name__ == "__main__":
    data_list = csv_to_list('./TrainAnnotations.csv')
    num_items = len(data_list)
    
    training_data_list = data_list[:num_items - num_validation_images]
    validation_data_list = data_list[-num_validation_images:]
    
    training_filepaths, training_labels = map(list, zip(*training_data_list))
    val_filepaths, val_labels = map(list, zip(*validation_data_list))


    training_label_dictionary = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[]
    }

    validation_label_dictionary = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[]
    }

    #Create a dictionary to separate training images into classes
    for index in range(len(training_filepaths)):
        training_label_dictionary[training_labels[index]].append(training_filepaths[index])
    
    for index in range(len(val_filepaths)):
        validation_label_dictionary[val_labels[index]].append(val_filepaths[index])

    #Copy images from validation set into directory structure
    for key in validation_label_dictionary.keys():
        image_list = validation_label_dictionary[key]
        for image_file in image_list:
            copy_file(im_path +  image_file, val_data_path + str(key) + '/')

    # Iterate over every key and create image tensors to flow over
    for key in training_label_dictionary.keys():
        print("Augmenting class " + str(key))
        filepath_list = training_label_dictionary[key]
        num_images = len(filepath_list)
        image_list = [ cv2.imread(im_path + path) for path in filepath_list ]
        images = np.array(image_list) # array of images
        images = np.reshape(images, (num_images, 480, 640, 3)) # tensor of images

        image_gen = ImageDataGenerator( # Select which will improve performance
            rescale=1./255,
            rotation_range=40,
            horizontal_flip=True,
            #vertical_flip=True,
            height_shift_range=0.15,
            width_shift_range=0.15,
            shear_range=0.15,
            brightness_range=[0.75,1.25],
            zoom_range=[0.75,1.0],
            fill_mode='nearest',
            #cval=0
        )

        image_gen.fit(images)

        #generate images and store them into directory specified by key
        tr_it = image_gen.flow(
            x=images, 
            save_to_dir=str(aug_data_path + str(key)),
            batch_size=batch_size,
            save_format='jpeg',
        )
        
        #create batches and save to directory
        for i in range(int(num_images_wanted/batch_size)):
            tr_it.next()
