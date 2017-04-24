import numpy as np 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split

batch_size = 10
num_classes = 2
epochs = 200

LEARNING_RATE = 1e-2

raw_data = np.loadtxt('training.txt')

labels = raw_data[:,-1]
labels = keras.utils.to_categorical(labels, num_classes)

data = raw_data[:,[0, 1, 2, 3]]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
adam = Adam(lr=LEARNING_RATE)
model.compile(loss='mse',optimizer=adam)
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
score_test = model.evaluate(X_test, y_test, verbose=0)
print score_test

def predict(model, new_data):
	preds = model.predict(new_data)
	return preds