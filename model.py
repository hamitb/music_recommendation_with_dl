from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import rmsprop
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import matplotlib as plt
import os
import numpy as np
import pandas as pd
import numpy as np
import pickle
import glob

num_classes = 7
image_size = 256
nb_epoch = 20


image_dir= 'images/'

images = []
genres = []

def scale_features(img):
  img /= 255.0
  return img

spect_files = glob.glob(image_dir + '*.png')
for file in spect_files:
  img = load_img('{}'.format(file), target_size=(256, 256))
  track_name = file.split('/')[2]
  genre = track_name.split('__')[0]
  img_array = img_to_array(img)
  img_array = scale_features(img_array)
  images.append(img_array)
  genres.append(genre)

# Conver genres to categorical
encoder = LabelBinarizer()
transformed_genres = encoder.fit_transform(genres)

# Split test and train sets
features_train, features_test, labels_train, labels_test = train_test_split(images, transformed_genres, test_size=0.3, random_state=42)
features_train, features_test = np.array(features_train), np.array(features_test)

# Create model
model = Sequential()


model.add(Conv2D(filters=64, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=128, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=256, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=512, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Flatten())
model.add(Dense(128))

model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
opt = rmsprop()

model.compile(loss='categorical_crossentropy',
             optimizer = opt,
             metrics = ['accuracy'])


# Fit
history = model.fit(features_train, labels_train, epochs=20, validation_data=(features_test, labels_test))

# Save
model.save('model.h5')
