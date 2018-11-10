#!/usr/bin/python3
# -*- coding:utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend
print("Using TensorFlow v{}".format(tf.__version__))

# Helper libraries
import numpy
import matplotlib.pyplot as plt
import time

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

LAYERS_SET = [ LAYER_1, LAYER_2, LAYER_3 ]

# https://keras.io/models/sequential/
model = keras.Sequential( LAYERS_SET )

# Compiling the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the train images
EPOCHS = 10

start_time = time.time()
model.fit(train_images, train_labels, epochs=EPOCHS)
end_time = time.time()
train_time_str = "Training time: {}s".format(end_time - start_time)
print(train_time_str)

# Evaluate the accuracy of the model with the train images
train_loss, train_acc = model.evaluate(train_images, train_labels)
train_loss_str = 'Train loss: {}%'.format(train_loss*100)
train_acc_str = 'Train accuracy: {}%'.format(train_acc*100)
print(train_loss_str)
print(train_acc_str)

# Evaluate the accuracy of the model with the test images
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc_str = 'Test accuracy: {}%'.format(test_acc*100)
test_loss_str = 'Test loss: {}%'.format(test_loss*100)
print(test_loss_str)
print(test_acc_str)

filename = 'deliverables/ver_1/result.txt'
with open(filename, 'w') as f_out:
    f_out.write(train_time_str + '\n')
    f_out.write(train_loss_str + '\n')
    f_out.write(train_acc_str + '\n')
    f_out.write(test_loss_str + '\n')
    f_out.write(test_acc_str + '\n')


# Predicting the labels of all train images
predictions = model.predict(train_images)
assigned_labels = numpy.argmax(predictions, axis=-1)
# assert len(predictions) == len(test_images)

filename = 'deliverables/ver_1/assigned_labels_train.txt'
with open(filename, 'w') as f_out:
    for label in assigned_labels:
        f_out.write(str(label))


# Predicting the labels of all test images
predictions = model.predict(test_images)
assigned_labels = numpy.argmax(predictions, axis=-1)

filename = 'deliverables/ver_1/assigned_labels_test.txt'
with open(filename, 'w') as f_out:
    for label in assigned_labels:
        f_out.write(str(label))
