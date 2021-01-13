#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import shutil
import random

import cv2
import matplotlib.pyplot as plt


def load_cascade(cascade_fp):
    return cv2.CascadeClassifier(cascade_fp)


def fix_file_ratio(split_files, split_path, eye_state, portion, total_amount, training_files, training_path):
    if len(split_files) < int(portion*total_amount):
        to_move = int(portion*total_amount - len(split_files))
        for file in random.sample(training_files, to_move):
            shutil.move(file, os.path.join(split_path, eye_state, os.path.split(file)[1]))
    else:
        to_move = int(len(split_files) - portion*total_amount)
        for file in random.sample(split_files, to_move):
            shutil.move(file, os.path.join(training_path, eye_state, os.path.split(file)[1]))


def plot_accuracy_and_loss(history):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].plot(
        range(1, len(history.history['accuracy']) + 1),
        history.history['accuracy'],
        'b-o', label='Training accuracy')
    ax[0].plot(
        range(1, len(history.history['val_accuracy']) + 1),
        history.history['val_accuracy'],
        'r-', label='Validation accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(
        range(1, len(history.history['loss']) + 1),
        history.history['loss'],
        'b-o', label='Training loss')
    ax[1].plot(
        range(1, len(history.history['val_loss']) + 1),
        history.history['val_loss'],
        'r-', label='Validation loss')
    ax[1].set_title('Loss')
    ax[1].legend()

    fig.show()