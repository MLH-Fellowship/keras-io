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

    # ================TESTING WITH COLOR WHEEL JPG =====================================================
    # filename = os.path.join(dirname + '/img/moving_square/color-wheel.jpg')
    # # loads image from file
    # img = cv2.imread(filename)
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (thresh, binaryImage) = cv2.threshold(grayImage, 249, 255, cv2.THRESH_BINARY)

    # # test to see if values are 0 or 255
    # # print (binaryImage[10,33])
    # # cv2.imshow('bw image', binaryImage)
    # path='./examples/vision/img/moving_square/bw-wheel2.jpg'
    # cv2.imwrite(path, binaryImage)

    # #uncomment to see image
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

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


def create_model():
    model = keras.Sequential()

    # shape should be this
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

    # model.compile(loss="binary_crossentropy", optimizer="adadelta", learning_rate=0.001)
    return model


# create_model()
# trouble-shooting
model1 = create_model()
opt = tf.keras.optimizers.Adadelta(
    learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta')
model1.compile(opt, loss="binary_crossentropy")


print(model1.name, model1.input_shape)

# for layer in model1.layers:
#     print(layer.name, layer.output_shape, layer.input_shape)


def getFrame(dirname, counter, row, col):
    filename = os.path.join(dirname + '/img/moving_square/frame_' + str(counter) + '.png')

    img = cv2.imread(filename)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, binaryImage) = cv2.threshold(grayImage, 0, 1, cv2.THRESH_BINARY)

    dim = (80, 80)
    resizedImage = cv2.resize(binaryImage, dim)
    return resizedImage


def generate_movies(n_samples=10, n_frames=60):
    rows = 80
    columns = 80
    movies = np.zeros((n_samples, n_frames, rows, columns, 1), dtype=np.float)  # features
    shifted_movies = np.zeros((n_samples, n_frames, rows, columns, 1), dtype=np.float)  # labels

    frame = 0
    # wrap in for-loop going over samples=10; have a bunch of identical instances in the batch;
    # make add noise func(after prediction works)
    while (frame < n_frames - 1):
        currentFrame = getFrame(dirname, counter, row, col)

    # add to dataset; need to put all the values from the [frame, x-val, y-val, binary value]
        for row in range(len(currentFrame)):
            for col in range(len(currentFrame[row])):
                movies[0, frame + 1, row, col] = currentFrame[row][col]
                shifted_movies[0, frame, row, col] = currentFrame[row][col]

        frame += 1


      # ==========AUGMENTATION===========================
    # create a loop that duplicates frames
    augmented_movies = np.zeros(
    (n_samples, n_frames, rows, columns, 1), dtype=np.float)  # duplicated frames

    dupeFrame = 0
    while (dupeFrame < 10000):
        currentFrame = getFrame(dirname, dupeFrame, rows, columns)
        for row in range(len(currentFrame)):
            for col in range(len(currentFrame[row])):
                augmented_movies[0, frame, row, col] = currentFrame[row][col]
        dupeFrame += 1

    # Cut to a 80x80 window
    movies = movies[::, ::, 20:100, 20:100, ::]
    shifted_movies = shifted_movies[::, ::, 20:100, 20:100, ::]
    augmented_movies= augmented_movies[::, ::, 20:100, 20:100, ::]

    movies[movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return movies, shifted_movies


# start training 100 copies without noise, if not perfect there's a problem. see if it memorizes.

# make identical batches, wrap in for loop


# ================= Train the model=========================================
# epochs = 1  # In practice, you would need hundreds of epochs.
epochs = 20
# need to figure out how to feed in the gif
noisy_movies, shifted_movies = generate_movies(n_frames=60)
print(np.array(noisy_movies).shape)


model1.fit(
    augmented_movies,  # features to predict on
    shifted_movies,  # labels
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
