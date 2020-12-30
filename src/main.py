import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn
from models import resnet
from utils.utils import *
import torchvision

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

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

show_random(images, emotion_nms_org= data['emotion_name'])
plt.show()

''' The following section is for image normalization.
The mean pixel intensity is calculated and subtracted to each image of the dataset.'''
pixel_mean = images.mean(axis=0)
pixel_std = np.std(images, axis=0)
images = np.divide(np.subtract(images, pixel_mean), pixel_std)

''' This section of code flattens the labels vector and counts how many classes the dataset has.
After that, it is created a one-hot vector to store each class as a 0 or a 1.'''
labels_flat = data['emotion'].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
#print(labels_count) # Low image reading can cause lower class count

# Flat vector with labels turned into matrix in which row represents a matrix.
# The '1' in each row represents the labeled emotion for that given image.
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)


'''Image reshaping and dataset splitting'''
# Reshaping and preparing image format for the model training.
images = images.reshape(images.shape[0], 48, 48, 1)
images = images.astype('float32')
train_loader = torch.utils.data.DataLoader(images, batch_size=16, shuffle=True, num_workers=0) # num_workers???


# Splitting images and labels into training, validation and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle = False)

#Note: if we are doing k cross validation then the val split is unnecessary and we need to substitute it.

''' This part of the code is for building the CNN model we are using  for the train'''
net = resnet.ResNet18()

criterion = nn.CrossEntropyLoss()
optim = 'Adam'
lr=1e-3  # Default learning rate
weight_decay=1e-4
epoch = 150
logspace = 0

if optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
elif optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
logspace_lr = torch.logspace(np.log10(lr), np.log10(lr) - logspace, epoch)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc_val = 0  # best test accuracy
best_acc_test = 0  # best test accuracy
start_epoch = 0  # start from epoch 0


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    loss_res = 0

    adjust_learning_rate(optimizer, epoch, lr)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        prob = F.softmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        loss.backward()
        # clip_grad_norm_(parameters=net.parameters(), max_norm=0.1)

        optimizer.step()

        train_loss += loss.item()
        loss_res = train_loss / (batch_idx + 1)

        max_num, predicted = prob.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return loss_res
#TODO: Do stratified k cross fold validation
#TODO: Build and compile model
#TODO: Try and test to see accuracy with resnet18
