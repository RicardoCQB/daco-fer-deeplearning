import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.utils import show_random
from sklearn.metrics import confusion_matrix


def plot_accuracy(data, size=(20, 10), model_name=''):
    figure.Figure(figsize=size)
    plt.plot(data['accuracy'])
    plt.plot(data['val_accuracy'])
    plt.title('Model Accuracy', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.legend(['Train', 'Val'], loc='upper left', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()
    plt.savefig('{}_accuracy.png'.format(model_name))


def plot_loss(data, size=(20, 10), model_name=''):
    figure.Figure(figsize=size)
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('Model Loss', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylim(0.9, 2)
    plt.legend(['Train', 'Val'], loc='upper left', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()
    plt.savefig('{}_loss.png'.format(model_name))



def predict_classes(model, test_imgs, test_labels, emotions_dict, batch_size=32):
    # Predict class of image using trained model
    class_pred = model.predict(test_imgs, batch_size=batch_size)

    # Convert vector of zeros and ones to label
    labels_pred = np.argmax(class_pred, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # Boolean array that indicates whether the predicted label is the true label
    correct = labels_pred == true_labels

    # Converting array of labels into emotion names
    pred_emotion_names = pd.Series(labels_pred).map(emotions_dict)

    results = {'Predicted_label': labels_pred, 'Predicted_emotion': pred_emotion_names, 'Is_correct': correct}
    results = pd.DataFrame(results)
    return correct, results


def visualize_predictions(images_test, orglabel_names, predlabel_names, correct_arr, valid=True, model_name=''):
    if valid == True:
        correct = np.array(np.where(correct_arr == True))[0]
        # Plot 15 randomly selected and correctly predicted images
        show_random(images_test, emotion_nms_org=orglabel_names, emotion_nms_pred=predlabel_names, random=False,
                    indices=correct)
        plt.show()
        plt.savefig('{}_correct_prediction_examples.png'.format(model_name))
    else:
        incorrect = np.array(np.where(correct_arr == False))[0]
        # Plot 15 randomly selected and wrongly predicted images
        show_random(images_test, emotion_nms_org=orglabel_names, emotion_nms_pred=predlabel_names, random=False,
                    indices=incorrect)
        plt.show()
        plt.savefig('{}_incorrect_prediction_examples.png'.format(model_name))


def create_confmat(true_labels, predicted_labels, columns, colour='Oranges', size=(20, 14), model_name=''):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm,
                         index=[col for col in columns],
                         columns=[col for col in columns])
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm_df, annot=True, cmap=colour, fmt='g', linewidths=.2)
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()
    plt.savefig('{}_confmat.png'.format(model_name))