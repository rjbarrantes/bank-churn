# ── Bank Churn Prediction ──
# Feedforward neural network for predicting customer churn in retail banking.
# Binary classification: will the customer leave (1) or stay (0)?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Data loading ──

# drop first 3 columns (row number, customer ID, surname) — no predictive value
churn = pd.read_csv('../../data/Churn_Modelling.csv')
churn = churn.iloc[:, 3:]

# features: credit score, geography, gender, age, tenure, balance,
#           num products, has credit card, is active, estimated salary
X = churn.iloc[:, :10]
y = churn.iloc[:, 10]


# ── Preprocessing ──

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# encode gender (binary) and one-hot encode geography (France/Germany/Spain)
X['Gender'] = LabelEncoder().fit_transform(X['Gender'])
X = pd.get_dummies(X, drop_first = True)

y = np.array(y)
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# ── Neural network ──
# progressive widening (8 → 16 → 32) lets the network learn increasingly
# complex feature interactions before collapsing to a single output node

from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, sigmoid
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

model = Sequential()
model.add(Dense(8, activation = relu))
model.add(Dense(16, activation = relu))
model.add(Dense(32, activation = relu))
model.add(Dense(1, activation = sigmoid))

model.compile(optimizer = Adam(), loss = binary_crossentropy, metrics = ['accuracy'])


# ── Training ──

model.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test))


# ── Evaluation ──

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > .5)

from sklearn.metrics import confusion_matrix

def plotConfusionMatrix(matrix):
    ax = sns.heatmap(matrix, cmap = 'Blues', annot = True, fmt = 'd')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

def plotLearningCurve(history, epoch):
    epochs = range(1, epoch + 1)

    plt.plot(epochs, history.history['accuracy'])
    plt.plot(epochs, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.show()

    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.show()

plotConfusionMatrix(confusion_matrix(y_test, y_pred_binary))
plotLearningCurve(model.history, 100)
