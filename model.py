import csv

import cv2
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_samples = samples[offset:end]

            images, measurements = [], []
            for batch_sample in batch_samples:
                # we do this to use the center camera and the side cameras
                for i in range(3):
                    source_path = batch_sample[i]
                    # source_path has backslashes because of Windows
                    filename = source_path.split('\\')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    if i == 1:
                        # left camera with correction
                        measurement = measurement + 0.2
                    if i == 2:
                        # right camera with correction
                        measurement = measurement - 0.2
                    measurements.append(measurement)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

input_shape = (160, 320, 3)
BATCH_SIZE = 32

# Model architecture
model = Sequential()
# Normalizing data
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=input_shape, output_shape=input_shape))
# crop images so we only have the important parts (the street)
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

# NVIDIA CNN
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.3))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples) / BATCH_SIZE),
                                  validation_data=validation_generator,
                                  validation_steps=int(len(validation_samples) / BATCH_SIZE),
                                  nb_epoch=5, verbose=1)
model.save('model.h5')
print('Saved model!')

plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('Mean squared error loss')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model.png')

print('Saved visualization data!')
