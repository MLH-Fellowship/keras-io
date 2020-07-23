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

""" seq = keras.Sequential(
    [
        keras.Input(
            shape=(None, 40, 40, 1)
        ),  # Variable-length sequence of 40x40x1 frames
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
) """


# libraries I'm adding


# really only changed the amount of frames in the shape




from tensorflow import keras
from tensorflow.keras import layers
from keras import models
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import os
import cv2
def create_model():
    model = keras.Sequential()

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         # 60 frames, 200x200 pi, 1 channel for black & white
                         input_shape=(60, 40, 40, 1),
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

    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model


"""
def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[
                    i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0
                ] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the model to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1) ** np.random.randint(0, 2)
                    noisy_movies[
                        i,
                        t,
                        x_shift - w - 1 : x_shift + w + 1,
                        y_shift - w - 1 : y_shift + w + 1,
                        0,
                    ] += (noise_f * 0.1)

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[
                    i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0
                ] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies
"""

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
    # joins directory and filename
    filename = os.path.join(
        dirname + '/img/moving_square/frame_18.png')

    # colored image to test
    # filename = os.path.join(dirname + '/img/moving_square/color-wheel.jpg')
    img = cv2.imread(filename)

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(
        grayImage, 0, 255, cv2.THRESH_BINARY)
    # loads image from file
    # img = cv2.imread(filename, 0)
    counter = 0

    # while (counter < n_frames):
    #     filename = os.path.join(dirname + '/img/moving_square/frame_'+ str(counter) +'.png')
    #     #0 stands for grayscale
    #     img = cv2.imread(filename, 0)
    #     counter = counter +1

    # display img
    # cv2.imshow('image'+ str(counter), img)

    cv2.imshow('image', grayImage)
    cv2.imshow('image', blackAndWhiteImage)
    print (blackAndWhiteImage[40,60])
    # need this or window automatically closes
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# colored image to test
    # filename = os.path.join(dirname + '/img/moving_square/color-wheel.jpg')
    # test filename
    #print('directory name  ' + dirname)
    #print('filename  ' + filename)
    # open filepath

    # start training 100 copies without noise, if not perfect there's a problem. see if it memorizes.

# test function call
generate_movies()


# planning to use the same training...
"""
# Train the model


epochs = 1  # In practice, you would need hundreds of epochs.
#need to figure out how to feed in the gif
noisy_movies, shifted_movies = generate_movies(n_samples=1200)
model.fit(
    noisy_movies[:1000],   #features to predict on
    shifted_movies[:1000],  #labels
    batch_size=10,
    epochs=epochs,
    verbose=2,
    validation_split=0.1,
)



# Test the model on one movie

Feed it with the first 7 positions and then
predict the new positions.

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
