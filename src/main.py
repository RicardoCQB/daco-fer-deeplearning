import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image
from utils.utils import show_random
#import torchvision.models as models


''' This section reads the dataset from the .csv file in the fer2013 folder '''
data_path_list = ["C:/Users/Ricardo/source/repos/daco-fer-deeplearning/data/fer2013/fer2013.csv", "/Users/esmeraldacruz/Documents/GitHub/daco-fer-deeplearning/data/fer2013/fer2013.csv","C:\\Users\\dtrdu\\Desktop\\Duarte\\Faculdade e Cadeiras\\DACO\\Project\\daco-fer-deeplearning\\data\\fer2013\\fer2013.csv", "C:/Users/Ricardo/source/daco-fer-deeplearning/data/fer2013/fer2013.csv"]
data=[]
data = pd.read_csv(data_path_list[3], nrows = 300)


''' The .csv file consists of the pixels of the 48x48 pixels image, and it also
has a label that determines each emotion and the purpose of the image (train or test) '''

# Atribution of each label to an emotion
emotions_dic = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

data['emotion_name'] = data['emotion'].map(emotions_dic)

im_pixel_values = data.pixels.str.split(" ").tolist()
im_pixel_values = pd.DataFrame(im_pixel_values, dtype=int)
images = im_pixel_values.values
images = images.astype(np.float)

#show_random(images, emotion_nms_org= data['emotion_name'])
#plt.show()

''' The following section is for image normalization.
The mean pixel intensity is calculated and subtracted to each image of the dataset.'''
pixel_mean = images.mean(axis=0)
pixel_std = np.std(images, axis=0)
images = np.divide(np.subtract(images, pixel_mean), pixel_std)

''' This section of code flattens the labels vector and counts how many classes the dataset has.
After that, it is created a one-hot vector to store each class count'''
labels_flat = data['emotion'].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
print(labels_count) # Low image reading can cause lower class count

# Function for creating zero/ones matrix indicating image label
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot


labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)




#TODO: Do stratified k cross fold validation
#TODO: Build and compile model
#TODO: Try and test to see accuracy with resnet18
