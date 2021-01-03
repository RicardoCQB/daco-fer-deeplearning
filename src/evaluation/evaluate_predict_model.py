import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model
from eval_pred_utils import *


#Loading the model and reading it's scores

model_name = ''
X_test = []
y_test = []

model = load_model(model_name)

scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=256)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

history = pd.read_csv('history.csv', usecols = ['acc','loss','val_acc','val_loss'])
plot_accuracy(history)
plot_loss(history)




