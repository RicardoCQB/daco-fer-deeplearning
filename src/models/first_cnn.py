''' This is the code of the first used CNN for this project'''

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model


def first_cnn():

    # Constructing CNN structure
    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1),
                     bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1),
                     bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                     kernel_initializer=RandomNormal(stddev=1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                     kernel_initializer=RandomNormal(stddev=1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # 5th convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                     kernel_initializer=RandomNormal(stddev=1)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                     kernel_initializer=RandomNormal(stddev=1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # 7th convolution layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                     kernel_initializer=RandomNormal(stddev=1)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                     kernel_initializer=RandomNormal(stddev=1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax')) # 7 is the number of classes

    return model
