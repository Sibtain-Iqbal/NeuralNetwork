import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categoricalz


with np.load("mnist.npz")  as data :
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]


type(x_train)

x_train.shape

img_1 = x_train[100]
x_train[500].shape

import matplotlib.pyplot as plt
plt.imshow(img_1)

print(x_train)


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

y_train[0]

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images to a 1D vector
model.add(Dense(511, activation='relu'))  # Fully connected layer with 512 neurons
model.add(Dense(256, activation='relu'))  # Fully connected layer with 256 neurons v  
model.add(Dense(10, activation='softmax'))  # Output layer with 10 units (one for each class)