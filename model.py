import csv

import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Convolution2D, Cropping2D
from keras.models import Sequential

lines = []
with open('./data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

images, measurements = [], []
for line in lines:
    # we do this to use the middle camera and the side cameras
    for i in range(3):
        source_path = line[i]
        # source_path has backslashes because of Windows
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

input_shape = X_train[0].shape

# Model architecture
model = Sequential()
# Normalizing data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
# crop images so we only have the important parts (the street)
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# NVIDIA CNN
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')

# TODO visualize model performance below
