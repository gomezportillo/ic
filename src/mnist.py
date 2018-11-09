#!/usr/bin/python3
# -*- coding:utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend
print("Using TensorFlow v{}".format(tf.__version__))

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Local libraries
from aux import render_image
from aux import render_imagelist

# Downloading MNIST dataset from keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Getting unique values from labels, sorting them and storing
class_names = list(set(train_labels))

# Prepocessing
## Normalising datasets before feeding to the neural network
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model
NODES_DENSE_1 = 128
NODES_DENSE_2 = 10

LAYER_1 = keras.layers.Flatten(input_shape=(28, 28))
LAYER_2 = keras.layers.Dense(NODES_DENSE_1, activation=tf.nn.relu)
LAYER_3 = keras.layers.Dense(NODES_DENSE_2, activation=tf.nn.softmax)

model = keras.Sequential([ LAYER_1, LAYER_2, LAYER_3 ])

# Compiling the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the train images
EPOCHS = 5
model.fit(train_images, train_labels, epochs=EPOCHS)

# Evaluate the accuracy of the model with the test images
test_loss, test_acc = model.evaluate(test_images, test_labels)
acc = 'Test accuracy: {}%'.format(test_acc*100)
print(acc)
