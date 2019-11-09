from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from allModelFile import *

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(f'tf ver :{tf.__version__}')


def build_model_without_keras(train_dataset):
    # TODO
    # y_hat = tf.constant(1, name='y_hat')
    # y = tf.constant(0, name='y')

    # loss = tf.Variable((y - y_hat) **2, name='loss')

    # init = tf.global_variables_initailizer
    pass


def build_model(train_dataset):
    #   model = keras.Sequential([
    #     layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(1, activation='sigmoid')
    #   ])

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    # loss = mse for normal regression
    # binary_crossentropy for logistic
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['mae', 'mse', 'categorical_accuracy', 'binary_accuracy'])
    return model


def load_data():
    matchData = readData("testMatches_noBool.csv")
    cleanedData = matchData.copy()

    train_dataset = cleanedData.sample(frac=0.8, random_state=0)
    test_dataset = cleanedData.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("radiant_win")
    train_stats = train_stats.transpose()

    train_dataset.pop('match_id')
    test_dataset.pop('match_id')

    train_labels = train_dataset.pop('radiant_win')
    test_labels = test_dataset.pop('radiant_win')
    return train_dataset, test_dataset, train_labels, test_labels


def train_model():
    train_dataset, test_dataset, train_labels, test_labels = load_data()

    model = build_model(train_dataset)

    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 10 == 0:
                print(f'{epoch//10}0')
            #if epoch % 10 == 0:
            print('.', end='')

    EPOCHS = 100
    # keras.callbacks.ProgbarLogger()
    history = model.fit(
        train_dataset, train_labels, batch_size=32,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot(), keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')],
        validation_data=(test_dataset, test_labels))

    loss, mae, mse, cat_accuracy, bin_accuracy  = model.evaluate(test_dataset, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
    print(f"loss: {loss}")
    print(f"mse: {mse}")
    print(f"cat_accuracy: {cat_accuracy}")
    print(f"bin_accuracy: {bin_accuracy}")
    # model_2 is normal linear regression, model_3 is for sigmoid with mse
    # model_4 is sigmoid with binary_crossentropy
    # model_5 is softmax
    model.save(r"model_4_test.h5")


def load_model():
    model = tf.keras.models.load_model("model_4_test.h5")
    return model


def nn_predict(model, heroList, index):
    train_dataset, test_dataset, train_labels, test_labels = load_data()
    # index = 1
    # print(f'type:{type(test_dataset)}')
    # print(f'test_dataset.iloc[0] : {test_dataset.iloc[0]}')
    # print(f'test_dataset.iloc[0].shape : {test_dataset.iloc[0].shape}')
    # print(f'transposing')
    # print(f'test_dataset.iloc[0].tranpose() : {test_dataset.iloc[0].transpose()}')
    # print(f'test_dataset.iloc[0].tranpose().shape : {test_dataset.iloc[0].transpose().shape}')
    test1 = np.array([test_dataset.iloc[index]])
    # print(f'test_dataset.iloc[0] : {test1}')
    # print(f'test_dataset.iloc[0].shape : {test1.shape}')
    result = model.predict(test1)
    # result = int(result > 0.5)
    print(f'result:{result}')
    print(f'int(result > 0.5):{int(result > 0.5)}')
    print(f'actual: {test_labels.iloc[index]}')


if __name__ == "__main__":
    train_model()
    # model = load_model()
    # nn_predict(model, '', 1)
    # for i in range(20, 30):
    #     nn_predict(model, '', i)
    #     print()
