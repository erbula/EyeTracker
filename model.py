#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob
import os
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D
from tensorflow.keras.layers import Dropout, Conv2D
from tensorflow.keras.models import Sequential

from utils import fix_file_ratio, plot_accuracy_and_loss


def create_model(input_shape):
    # create the model here below
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="softmax"))

    model.compile(
        optimizer='adam', #RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_generator(with_augmentation=True):
    if with_augmentation:
        data_generator = ImageDataGenerator(
            #preprocessing_function=preprocess_input,   # add if needed
            rescale=1.0/255,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05
        )
    else:
        data_generator = ImageDataGenerator(
            #preprocessing_function=preprocess_input,   # add if needed
            rescale=1.0/255
        )

    return data_generator


class ModelEntity:

    def __init__(self, model=None):
        self.model = model
        self.path_train = './data/train'
        self.path_valid = './data/valid'
        self.path_test = './data/test'
        self.total_file_amount = None
        self.history = None

    def split_data(self, valid_portion, test_portion):
        states = ['open', 'closed']

        # making sure the destination folders exist, make them if not
        for state in states:
            os.makedirs(os.path.join(self.path_valid, state), exist_ok=True)
            os.makedirs(os.path.join(self.path_test, state), exist_ok=True)

            train_files = glob.glob(os.path.join(self.path_train, state, '*.bmp'))
            valid_files = glob.glob(os.path.join(self.path_valid, state, '*.bmp'))
            test_files = glob.glob(os.path.join(self.path_test, state, '*.bmp'))
            self.total_file_amount = len(train_files) + len(valid_files) + len(test_files)

            # checking the current portion of validation data, fixing it according to valid_portion
            fix_file_ratio(
                valid_files, self.path_valid, state, valid_portion,
                self.total_file_amount, train_files, self.path_train
            )
            # re-check train_files
            train_files = glob.glob(os.path.join(self.path_train, state, '*.bmp'))
            # checking the current portion of test data, fixing it according to test_portion
            fix_file_ratio(
                test_files, self.path_test, state, test_portion,
                self.total_file_amount, train_files, self.path_train
            )

    def set_up_generators(self, target_size, batch_size, color_mode):
        train_data_generator = create_generator()
        valid_data_generator = create_generator()
        test_data_generator = create_generator(False)

        self.train_datagen = train_data_generator.flow_from_directory(
            directory=self.path_train,
            target_size=target_size,
            color_mode=color_mode,
            class_mode='binary',
            batch_size=batch_size,
            shuffle=True,
            seed=13
        )

        self.valid_datagen = valid_data_generator.flow_from_directory(
            directory=self.path_valid,
            target_size=target_size,
            color_mode=color_mode,
            class_mode='binary',
            batch_size=batch_size,
            shuffle=True,
            seed=13
        )

        self.test_datagen = test_data_generator.flow_from_directory(
            directory=self.path_test,
            target_size=target_size,
            color_mode=color_mode,
            class_mode='binary',
            batch_size=batch_size,
            shuffle=False
        )

    def train(self, epochs, batch_size):
        if self.model is None:
            raise Exception('Error! No model attached!')

        checkpoint = ModelCheckpoint(
            f'./model/{round(time.time())}.tf',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch'
        )

        self.history = self.model.fit(
            self.train_datagen,
            epochs=epochs,
            steps_per_epoch=self.train_datagen.samples//batch_size,
            validation_data=self.valid_datagen,
            validation_steps=self.valid_datagen.samples//batch_size,
            callbacks=[checkpoint]
        )

    def evaluate_model(self):
        print(self.model.evaluate(self.test_datagen))


if __name__ == '__main__':
    # example
    BATCH = 32
    TARGET_SIZE = (256, 256, 3)

    me = ModelEntity()
    me.split_data(0.25, 0.25)
    me.set_up_generators(TARGET_SIZE[:2], BATCH, 'rgb')
    me.model = create_model(TARGET_SIZE)
    me.train(epochs=10, batch_size=BATCH)

    plot_accuracy_and_loss(me.history)
    me.evaluate_model()