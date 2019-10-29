# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# read data

churn = pd.read_csv('../../data/Churn_Modelling.csv')
churn = churn.iloc[:, 3:]
churn.head()


# define X and y

X = churn.iloc[:, :10]
y = churn.iloc[:, 10]


# encode categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
X['Gender'] = LabelEncoder().fit_transform(X['Gender'])
X = pd.get_dummies(X, drop_first = True)


# preview data for algorithm

print(X.head())
print(y.head())


# convert X and y to arrays for nn

y = np.array(y)
X = np.array(X)


# split into train and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# scale features

from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# build neural network

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


# fit model

model.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test))


# make predictions on test set

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > .5)


# model accuracy visualizations

from sklearn.metrics import confusion_matrix

def plotConfusionMatrix(matrix):
    ax = sns.heatmap(matrix, cmap = 'Blues', annot = True, fmt = 'd')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

def plotLearningCurve(history, epoch):
    
    # plot training & validation accuracy values
    epochs = range(1, epoch + 1)
    plt.plot(epochs, history.history['accuracy'])
    plt.plot(epochs, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.show()
    
    # plot training & validation loss values
    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.show()


plotConfusionMatrix(confusion_matrix(y_test, y_pred_binary))
plotLearningCurve(model.history, 100)