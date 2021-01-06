import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.utils import dense_to_one_hot

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import tensorflow as tf


''' This section reads the dataset from the .csv file in the fer2013 folder '''
data_path_list = ["C:/Users/Ricardo/source/repos/daco-fer-deeplearning/data/fer2013/fer2013.csv", "/Users/esmeraldacruz/Documents/GitHub/daco-fer-deeplearning/data/fer2013/fer2013.csv","C:\\Users\\dtrdu\\Desktop\\Duarte\\Faculdade e Cadeiras\\DACO\\Project\\daco-fer-deeplearning\\data\\fer2013\\fer2013.csv", "C:/Users/Ricardo/source/daco-fer-deeplearning/data/fer2013/fer2013.csv"]
data=[]
num_images_to_read = 150
print('Reading data now...')
data = pd.read_csv(data_path_list[0], nrows=num_images_to_read)
#data = pd.read_csv(data_path_list[0])


''' The .csv file consists of the pixels of the 48x48 pixels image, and it also
has a label that determines each emotion and the purpose of the image (train or test) '''

# Atribution of each label to an emotion
emotions_names = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

data['emotion_name'] = data['emotion'].map(emotions_names)

im_pixel_values = data.pixels.str.split(" ").tolist()
im_pixel_values = pd.DataFrame(im_pixel_values, dtype=int)
images = im_pixel_values.values
images = images.astype(np.float)

#test_idx_start = int(num_images_to_read*0.9)
test_idx_start = 32298
images_test = images[test_idx_start:]

''' The following section is for image normalization.
The mean pixel intensity is calculated and subtracted to each image of the dataset.'''
pixel_mean = images.mean(axis=0)
pixel_std = np.std(images, axis=0)
images = np.divide(np.subtract(images, pixel_mean), pixel_std)

''' This section of code flattens the labels vector and counts how many classes the dataset has.
After that, it is created a one-hot vector to store each class as a 0 or a 1.'''
labels_flat = data['emotion'].values.ravel()
labels_count = np.unique(labels_flat).shape[0]

# Flat vector with labels turned into matrix in which row represents a matrix.
# The '1' in each row represents the labeled emotion for that given image.
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

'''Image reshaping and dataset splitting'''
# Reshaping and preparing image format for the model training.
images = images.reshape(images.shape[0], 48, 48, 1)

for im in images:
    im[:][:][1] = im[:][:][0]
    im[:][:][2] = im[:][:][0]

print(images[0].shape)

images = images.astype('float32')