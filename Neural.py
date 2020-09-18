import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
import time
import glob
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(816)
device = tf.test.gpu_device_name()
tf.keras.backend.clear_session()  # For easy reset of notebook state.

input_layer = keras.Input(shape=(100, 20, 1), name='main_input')
x = layers.Conv2D(16, 8, padding='same', activation='relu')(input_layer)
x = layers.Conv2D(32, 6, padding='same', activation='relu')(x)
x = layers.Conv2D(8, 4, padding='same', activation='relu')(x)
x = layers.Conv2D(4, 2, padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(8)(x)
x = layers.Dense(64)(x)
output = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=input_layer, outputs=output)

model.summary()

# compiling the model
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)  # default params
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# 获取一位受试者的训练数据
train_data = np.array(joblib.load('data/event_data_by_S.pkl')[0])
train_event = np.array(joblib.load('data/event_labels_by_S.pkl')[0])

reshaped_data = []

for i in range(len(train_data)):
    reshaped_data.append(train_data[i].reshape(2000,))

reshaped_data = np.array(reshaped_data)

train_event = to_categorical(train_event)


def train_net(model):
    init = time.time()

    X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_event, test_size=0.1, random_state=816)
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=30, epochs=75, verbose=1)
    end = time.time()
    print("time elapsed training is:", (end - init)/60, " minutes")
    return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss']


acc, val_acc, loss, val_loss = train_net(model)
