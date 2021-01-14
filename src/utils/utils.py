import numpy as np
import matplotlib.pyplot as plt



# Function for displaying 15 random images
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense


def show_random(imgs, emotion_nms_org = None, emotion_nms_pred = None, random = True, indices = None):
    """

    Function displaying 15 randomly chosen images. Arguments:

    imgs:  Source of images

    emotion_nms_org: Default "None", if specified, should be a Pandas Series object consisting of emotion names. As a result, emotion name will be displayed above every image.

    emotion_nms_pred: Default "None", if specified should be a Pandas Series object with predicted emotion names. As a result, emotion name will be displayed above image.

    random: Defult "True", indices will be randomly drawn from “discrete uniform” distribution starting at 0 up to max(len(imgs) otherwise randomly chosen from values passed into "indices" argument without replacement.

    indices: Default "None", if specified "random" should be set to "False" to draw random images from the variable passed into "indices" argument starting at min(len(indices)) up to max(len(indices)) and not using "discrete uniform" distribution.

    """

    if random == True:
        indices = np.random.randint(0, len(imgs), size=15)
    else:
        print(len(indices))
        indices = np.random.choice(list(indices), size=15, replace=False)
    plt.figure(figsize=(20, 14))
    for index, number in enumerate(indices):
        plt.subplot(3 ,5, index + 1)
        if (isinstance(emotion_nms_org, type(None)) & isinstance(emotion_nms_pred, type(None))):
            plt.title('Image: ' + str(indices[index]))
        elif (isinstance(emotion_nms_org, type(None)) & ~isinstance(emotion_nms_pred, type(None))):
            plt.title('Image: ' + str(indices[index]) + '\n' + 'Predicted emotion:' + emotion_nms_pred[indices[index]])
        elif (~isinstance(emotion_nms_org, type(None)) & isinstance(emotion_nms_pred, type(None))):
            plt.title('Image: ' + str(indices[index]) + '\n' + 'Original emotion: ' + emotion_nms_org[indices[index]])
        else:
            plt.title('Image: ' + str(indices[index]) + '\n' + 'Original emotion: ' + emotion_nms_org[indices[index]] +
                      '\n' + 'Predicted emotion:' + emotion_nms_pred[indices[index]])
        show_image = imgs[number].reshape(48 ,48)
        plt.axis('off')
        plt.imshow(show_image, cmap='gray')

# Function for creating zero/ones matrix indicating image label
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot


# Function to transform grayscale 4d tensor of images to RGV 4d tensor
def grayscale_tensor_toRGB(images, num_images):
    images2 = np.zeros((num_images, 48, 48, 3))

    for i, image in enumerate(images2):
        for j, pix_row in enumerate(image):
            for k, pixel in enumerate(pix_row):
                for m, channel in enumerate(pixel):
                    images2[i][j][k][m] = images[i][j][k][0]

    images2 = images2.astype('float32')


# Function to add the fully connected layer to keras applications models
def add_fc_layer(model, labels_count):
    model = Sequential()
    model.add(model)
    model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Dense(labels_count, activation='softmax'))