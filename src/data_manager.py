import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image

''' This section reads the dataset from the .csv file in the fer2013 folder '''
data_path_list = ["C:/Users/Ricardo/source/repos/daco-fer-deeplearning/data/fer2013/fer2013.csv", "/Users/esmeraldacruz/Documents/GitHub/daco-fer-deeplearning/data/fer2013/fer2013.csv","C:\\Users\\dtrdu\\Desktop\\Duarte\\Faculdade e Cadeiras\\DACO\\Project\\daco-fer-deeplearning\\data\\fer2013\\fer2013.csv" ]
data=[]
data = pd.read_csv(data_path_list[0], nrows = 10)


''' The .csv file consists of the pixels of the 48x48 pixels image, and it also
has a label that determines each emotion and the purpose of the image (train or test) '''

#Atribution of each label to an emotion
emotions_dic = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

data['emotion_name'] = data['emotion'].map(emotions_dic)

im_pixel_values = data.pixels.str.split(" ").tolist()
im_pixel_values = pd.DataFrame(im_pixel_values, dtype=int)
images = im_pixel_values.values
images = images.astype(np.float)

im_test = images[1].reshape(48, 48)
im_test = Image.fromarray(im_test) 
im_test.show()