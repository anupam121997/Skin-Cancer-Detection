# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:05:56 2020

@author: LENOVO
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model,save_model
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense,ActivityRegularization
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import tensorflow
import os
import keras
from tensorflow import lite
from keras.preprocessing.image import array_to_img
from time import strftime
from keras.callbacks import TensorBoard

image_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

image_gen.flow_from_directory('DermMel/train_sep/')

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),
        input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),
        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),
        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),
        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(ActivityRegularization(l2 = 0.001))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer=keras.optimizers.Adam(learning_rate = 1e-4),
             metrics=['accuracy'])



model.summary()

batch_size=20

train_image_gen=image_gen.flow_from_directory('DermMel/train_sep/',
                                              target_size=(150,150),
                                             batch_size=batch_size,
                                              class_mode='binary')




validation_image_gen=image_gen.flow_from_directory('DermMel/valid/',
                                                  target_size=(150,150),
                                                  batch_size=batch_size,
                                                  class_mode='binary')


train_image_gen.class_indices
"""
history = model.fit_generator(train_image_gen,
                              epochs=100,
                              steps_per_epoch = 10682 // batch_size,
                              validation_data = validation_image_gen,
                              validation_steps = 3562 // batch_size)

model.save('malenoma.h5')

model.save_weights('malenoma_weights.h5')
"""
model2=load_model('malenoma.h5')

image_gen.flow_from_directory('DermMel/test/')
test_image_gen=image_gen.flow_from_directory('DermMel/test/',
                                              target_size=(150,150),
                                             batch_size=batch_size,
                                             class_mode='binary')

score = model2.evaluate(test_image_gen)
