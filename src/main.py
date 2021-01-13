
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import GlobalAveragePooling2D

from utils.utils import *
from evaluation.eval_pred_utils import *
from cnn_models import first_cnn

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import tensorflow as tf

''' This section reads the dataset from the .csv file in the fer2013 folder '''
data_path_list = ["C:/Users/Ricardo/source/repos/daco-fer-deeplearning/data/fer2013/fer2013.csv", "/Users/esmeraldacruz/Documents/GitHub/daco-fer-deeplearning/data/fer2013/fer2013.csv","C:\\Users\\dtrdu\\Desktop\\Duarte\\Faculdade e Cadeiras\\DACO\\Project\\daco-fer-deeplearning\\data\\fer2013\\fer2013.csv", "C:/Users/Ricardo/source/daco-fer-deeplearning/data/fer2013/fer2013.csv"]
data = []
num_images_to_read = 150
print('Reading data now...')
#data = pd.read_csv(data_path_list[3], nrows=num_images_to_read)
data = pd.read_csv(data_path_list[0])


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
num_images = images.shape[0]
images = images.reshape(num_images, 48, 48, 1)

images2 = np.zeros((num_images, 48, 48, 3))

for i, image in enumerate(images2):
    for j, pix_row in enumerate(image):
        for k, pixel in enumerate(pix_row):
            for m, channel in enumerate(pixel):
                images2[i][j][k][m] = images[i][j][k][0]

images2 = images2.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(images2, labels, test_size=0.1, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle = False)

''' This part of the code is for building the CNN model we are using  for the train'''
# Constructing CNN structure
resnet_model = tf.keras.applications.ResNet50(input_shape=(48, 48, 3), classes=labels_count,
                                       weights='imagenet', include_top=False)
model = Sequential()
model.add(resnet_model)
model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(Dense(labels_count, activation='softmax'))

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()

# Specifying parameters for Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
    zoom_range=0.05)  # zoom images in range [1 - zoom_range, 1+ zoom_range]

datagen.fit(X_train)

# Saving model each time it achieves lower loss on the validation set
filepath = 'Resnet50_150_Epochs.hdf5'
history_filepath = "{}_history.csv".format(filepath)
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir='./logs')

history_object = model.fit(datagen.flow(X_train, y_train,
                    batch_size=32),
                    epochs=150,
                    validation_data=(X_val, y_val),
                    steps_per_epoch=X_train.shape[0]/32,
                    callbacks=[checkpointer, tensorboard]),

pd.DataFrame(history_object[0].history).to_csv(history_filepath)

#Loading the model and reading it's scores

model_name = filepath
model_loaded = load_model(model_name)

scores = model_loaded.evaluate(np.array(X_test), np.array(y_test), batch_size=1024)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

history = pd.read_csv(history_filepath, usecols = ['loss','accuracy','val_loss','val_accuracy'])
plot_accuracy(history, model_name=model_name)
plot_loss(history, model_name=model_name)

correct, results_df = predict_classes(model_loaded, X_test, y_test, emotions_names, batch_size=1024)
results_df['Original_label'] = data['emotion'][test_idx_start:].values
results_df['True_emotion'] = results_df['Original_label'].map(emotions_names)

create_confmat(results_df['Original_label'], results_df['Predicted_label'],
               ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'], colour='Greens')

visualize_predictions(images_test, results_df['True_emotion'], results_df['Predicted_emotion'], correct,
                      valid=True, model_name=model_name)

visualize_predictions(images_test, results_df['True_emotion'], results_df['Predicted_emotion'], correct,
                      valid=False, model_name=model_name)



#TODO: Do stratified k cross fold validation

