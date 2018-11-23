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
num_classes = len(class_names)
input_shape = (28, 28, 1) # 28x28 pixels, no z coordinate

# Printing how manny different numbers are in each set
print_unique_numbers('train', train_labels)
print_unique_numbers('test', test_labels)

# Prepocessing
## Normalising datasets before feeding to the neural network
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32')

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images = test_images.astype('float32')

## convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# Building the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(4, 4),
                 activation=tf.nn.relu,
                 input_shape=input_shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(248, activation=tf.nn.relu))
model.add(keras.layers.Dense(124, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(num_classes, activation=tf.nn.softmax))

# Compiling the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# Train the model with the train images
EPOCHS = 20

train_time_str = train_model(model, train_images, train_labels, EPOCHS)

# Evaluate the accuracy of the model with the train images
train_loss_str, train_acc_str = evaluate_model('Train', model, train_images, train_labels)

# Evaluate the accuracy of the model with the test images
test_loss_str, test_acc_str = evaluate_model('Test', model, test_images, test_labels)

# Save results
VERSION = 3
save_results(VERSION, train_time_str, train_loss_str, train_acc_str, test_loss_str, test_acc_str)

# Predicting the labels of all train images
predict_labels(VERSION, 'train', model, train_images)

# Predicting the labels of all test images
predict_labels(VERSION, 'test', model, test_images)

save_model_layers_to_file(VERSION, 'model.png', model)
