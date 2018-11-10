#!/usr/bin/python3
# -*- coding:utf-8 -*-

import time
import os
import numpy

def train_model(model, images, labels, EPOCHS):
    start_time = time.time()

    model.fit(images, labels, epochs=EPOCHS)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = "Training time: {}s".format(elapsed_time)
    print(elapsed_time_str)

    return elapsed_time_str

def evaluate_model(TYPE, model, images, labels):
    loss, acc = model.evaluate(images, labels)
    loss_str = '{0} loss: {1}%'.format(str(TYPE), loss*100)
    acc_str = '{0} accuracy: {1}%'.format(str(TYPE), acc*100)
    print(loss_str)
    print(acc_str)

    return (loss_str, acc_str)

def save_results(VERSION, train_time, train_loss, train_acc, test_loss, test_acc):
    dir_name = 'deliverables/ver_{0}/'.format(VERSION)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    filename = dir_name + 'result.txt'
    with open(filename, 'w') as f_out:
        f_out.write(train_time + '\n')
        f_out.write(train_loss + '\n')
        f_out.write(train_acc + '\n')
        f_out.write(test_loss + '\n')
        f_out.write(test_acc + '\n')

def predict_labels(VERSION, TYPE, model, images):
    predictions = model.predict(images)
    assigned_labels = numpy.argmax(predictions, axis=-1)

    assert len(predictions) == len(images)

    filename = 'deliverables/ver_{0}/assigned_labels_{1}.txt'.format(VERSION, TYPE)
    with open(filename, 'w') as f_out:
        for label in assigned_labels:
            f_out.write(str(label))
