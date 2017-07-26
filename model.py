import csv
from pathlib import Path
from random import shuffle

import cv2
import matplotlib
import sklearn
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
from keras.models import Sequential, model_from_json
import json
import matplotlib.image as mpimg

lines = []
with open('./data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images, measurements = [], []
            # we do this to use the center camera and the side cameras
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[0]
                    # source_path has backslashes because of Windows
                    filename = source_path.split('\\')[-1]
                    current_path = './data/IMG/' + filename
                    image = mpimg.imread(current_path)
                    image = preprocess_img(image)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    correction = 0.3
                    if i == 1:
                        # left camera with correction
                        measurement += correction
                    if i == 2:
                        # right camera with correction
                        measurement -= correction
                    measurements.append(measurement)
                    # augment data
                    images.append(np.fliplr(image))
                    measurements.append(measurement * -1.0)

            x_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(x_train, y_train)


def nvidia_cnn(dropout=0.3, epochs=3, batch_size=32):
    input_shape = (40, 160, 3)
    # Model architecture
    model = Sequential()
    # Normalizing data
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=input_shape, output_shape=input_shape))

    # NVIDIA CNN
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.summary()

    model.compile(loss='mse', optimizer='adam')

    if Path("model.json").is_file() and Path("model.h5").is_file():
        model = load_trained_model()

    checkpoint = ModelCheckpoint(filepath='model.h5', save_best_only=True, monitor='val_loss')
    history_obj = model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples) / batch_size),
                                      validation_data=validation_generator,
                                      validation_steps=int(len(validation_samples) / batch_size),
                                      epochs=epochs, callbacks=[checkpoint])

    # Save model
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)
    print('Saved model!')

    # Save model data visualization
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('Mean squared error loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('model.png')
    print('Saved visualization data!')


def preprocess_img(image):
    # resize to half of the original size (320x160 to 160x80)
    h, w = image.shape[:2]
    ratio = (w * 0.5) / w
    dim = (int(w * 0.5), int(h * ratio))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # crop image to 160x40 (take away 30px from top and 10px from bottom) so we only have the important parts
    image = image[30:70, 0:160]
    return image


def load_trained_model():
    model = model_from_json(open('model.json').read())
    # load weights into new model
    model.load_weights("model.h5")
    model.compile(loss='mse', optimizer='adam')
    return model


BATCH_SIZE = 64
EPOCHS = 5
DROPOUT = 0.5

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

nvidia_cnn(dropout=DROPOUT, epochs=EPOCHS, batch_size=BATCH_SIZE)
