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