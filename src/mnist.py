#!/usr/bin/python3
# -*- coding:utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend
print("Using TensorFlow v{}".format(tf.__version__))

#Local libraries
from aux import *
from mnist_functions import *

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

model = keras.Sequential()
model.add( LAYER_1 )
model.add( LAYER_2 )
model.add( LAYER_3 )

# Compiling the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the train images
EPOCHS = 15

train_time_str = train_model(model, train_images, train_labels, EPOCHS)

# Evaluate the accuracy of the model with the train images
train_loss_str, train_acc_str = evaluate_model('Train', model, train_images, train_labels)


# Evaluate the accuracy of the model with the test images
test_loss_str, test_acc_str = evaluate_model('Test', model, test_images, test_labels)

# Save results
VERSION = 1
save_results(VERSION, train_time_str, train_loss_str, train_acc_str, test_loss_str, test_acc_str)

# Predicting the labels of all train images
predict_labels(VERSION, 'Train', model, train_images)

# Predicting the labels of all test images
predict_labels(VERSION, 'Test', model, test_images)
