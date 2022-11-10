import keras;
from keras.models import Sequential;
from keras.layers import *;
from ann_visualizer.visualize import ann_viz

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/65840/.conda/envs/tf/Lib/site-packages/graphviz/backend/'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def build_cnn_model():
    model = keras.models.Sequential()

    model.add(
          Conv2D(
              32, (3, 3),
              padding="same",
              input_shape=(32, 32, 3),
              activation="relu"))
    model.add(Dropout(0.2))

    model.add(
          Conv2D(
              32, (3, 3),
              padding="same",
              input_shape=(32, 32, 3),
              activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(
          Conv2D(
              64, (3, 3),
              padding="same",
              input_shape=(32, 32, 3),
              activation="relu"))
    model.add(Dropout(0.2))

    model.add(
          Conv2D(
              64, (3, 3),
              padding="same",
              input_shape=(32, 32, 3),
              activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation="softmax"))

    return model

model = build_cnn_model()
ann_viz(model, view=True, filename='temp.gv')