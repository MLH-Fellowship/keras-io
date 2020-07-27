"""
Title: Next-frame prediction with Conv-LSTM
Author: [jeammimi](https://github.com/jeammimi)
Date created: 2016/11/02
Last modified: 2020/05/01
Description: Predict the next frame in a sequence using a Conv-LSTM model.
"""
"""
# Introduction

This script demonstrates the use of a convolutional LSTM model.
The model is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""

"""
# Setup
"""


"""
# Build a model

We create a model which take as input movies of shape
`(n_frames, width, height, channels)` and returns a movie
of identical shape.
"""

import os
import pylab as plt
import numpy as np
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv3D
from tensorflow import keras
from tensorflow.keras import layers
from keras import models
from cv2 import cv2  # gets rid of pylint errors
"""
# editted input shape of the original code
def create_model():
    seq = keras.Sequential(
        [
            keras.Input(
                shape=(None, 40, 40, 1 )
            ),
            # layers.ConvLSTM2D(
            #     filters=60, kernel_size=(3, 3), padding="same", return_sequences=True
            # ),
            # layers.BatchNormalization(),

            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            ),
        ]
    )
    seq.compile(loss="binary_crossentropy", optimizer="adadelta")

    return seq
"""
def create_model():
    model = keras.Sequential()
    # model = Sequential()

    model.add(keras.Input(shape=(None, 40, 40, 1)))
                         # 60 frames, 200x200 pi, 1 channel for black & white

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))
    model.compile(loss="binary_crossentropy", optimizer="adadelta")
    return model


# create_model()
for layer in create_model().layers:
    print(layer.output_shape)
# frame generation

# only have one sample


def generate_movies(n_frames=60):
    row = 80
    col = 80
    # features
    noisy_movies = np.zeros((n_frames, row, col, 1), dtype=np.float)
    # keep this for labels
    shifted_movies = np.zeros((n_frames, row, col, 1), dtype=np.float)

    # use relative filepath to grab the frames
    # returns directory path of file
    dirname = os.path.dirname(__file__)

    # ================TESTING WITH COLOR WHEEL JPG =====================================================
    # filename = os.path.join(dirname + '/img/moving_square/color-wheel.jpg')
    # # loads image from file
    # img = cv2.imread(filename)
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 249, 255, cv2.THRESH_BINARY)

    # # test to see if values are 0 or 255
    # # print (blackAndWhiteImage[10,33])
    # # cv2.imshow('bw image', blackAndWhiteImage)

    # path='./examples/vision/img/moving_square/bw-wheel2.jpg'
    # cv2.imwrite(path, blackAndWhiteImage)

    # #uncomment to see image
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

# ====================ONCE TEST IMAGE IS WORKING, USE BELOW=========================================
    counter = 0

    while (counter < n_frames):
        filename = os.path.join(
            dirname + '/img/moving_square/frame_' + str(counter) + '.png')

        # 0 stands for grayscale
        img = cv2.imread(filename)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(
            grayImage, 0, 255, cv2.THRESH_BINARY)
        # save B&W image
        cv2.imwrite('./examples/vision/img/moving_square/bw_frame_' +
                    str(counter) + '.png', blackAndWhiteImage)
        # blackAndWhiteImage.save('/img/moving_square/frameee_' + str(counter)+ '.png')
        counter += 1

    # display img
    # cv2.imshow('bw image'+ str(counter), blackAndWhiteImage)

#    # need this or window automatically closes
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies
    # start training 100 copies without noise, if not perfect there's a problem. see if it memorizes.


# test function call
# generate_movies(n_frames)


# planning to use the same training...

# Train the model


# epochs = 1  # In practice, you would need hundreds of epochs.
epochs = 1
# need to figure out how to feed in the gif
noisy_movies, shifted_movies = generate_movies(n_frames=60)
create_model().fit(
    noisy_movies[:60],  # features to predict on
    shifted_movies[:60],  # labels
    batch_size=10,
    epochs=epochs,
    verbose=2,
    validation_split=0.1,
)


# Test the model on one movie
"""
Feed it with the first 7 positions and then
predict the new positions.
"""
"""
movie_index = 1004
test_movie = noisy_movies[movie_index]

# Start from first 7 frames
track = test_movie[:7, ::, ::, ::]

# Predict 16 frames
for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
"""
