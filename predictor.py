# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:58:28 2019

@author: ZY
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import math

print('finish import')
def getData(training=True):
    all_data = pd.read_csv('testMatches_noBool.csv', error_bad_lines=False)
    all_data = all_data.drop(['match_id'], axis=1)
    if training:
        train = all_data[::2]
        test = all_data[1::2]
        print('data obtained')
        return train, test
    else:
        return all_data

def saveModel():
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

    inc_v1 = v1.assign(v1 + 1)
    dec_v2 = v2.assign(v2 - 1)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.
        inc_v1.op.run()
        dec_v2.op.run()
        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

def predict(radiant, dire, classifier):
    print('starting prediction')
    data = np.zeros(258, dtype=int)
    for i in radiant:
        data[int(i)] = 1
    for i in dire:
        data[int(i) + 129] = 1

    data = np.array([data]).astype(np.float32)
    y_out = list(classifier.predict(x=data))
    print('prediction completed')
    out_hard = list(y['classes'] for y in y_out)
    out_soft = list(y['probabilities'] for y in y_out)

    # SElf-note
    # Soft predictions [dire win, radiant win]
    print("Soft predictions:")
    print(out_soft[:5])
    print("Hard predictions:")
    print(out_hard[:5])
    # return radiant win percentage
    return out_soft[0][1]

def createClassifier():
    params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
      num_classes=2, num_features=258,
      num_trees=500, max_nodes=1000).fill()
    print('params set')
    # params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    #  num_classes=3, num_features=4, num_trees=50, max_nodes=1000, split_after_samples=50).fill()

    # Generate Classifier
    # for f in glob("./*"):
    #     os.remove(f)
    classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
        params, model_dir="./")

    # Training
    train = getData(training=False)
    # print(type(test['radiant_win'][1]))
    x_train = train.drop(['radiant_win'], axis=1).astype(np.float32).values
    print(type(x_train), x_train)
    #label_map = {'False': 0, 'True': 1}
    #label_map = {'Radiant': 1, 'Dire': 0}
    y_train = train['radiant_win'].astype(np.float32).values
    print(f'y_train:\n {y_train}')
    print('fitting time')
    classifier.fit(x=x_train, y=y_train)
    return classifier

if __name__ == "__main__":
    params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
      num_classes=2, num_features=258,
      num_trees=50, max_nodes=1000).fill()
    print('params set')
    # params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    #  num_classes=3, num_features=4, num_trees=50, max_nodes=1000, split_after_samples=50).fill()

    # Generate Classifier
    # for f in glob("./*"):
    #     os.remove(f)
    classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
        params, model_dir="./")

    # Training
    train, test = getData()
    # print(type(test['radiant_win'][1]))
    x_train = train.drop(['radiant_win'], axis=1).astype(np.float32).values
    print(type(x_train), x_train)
    #label_map = {'False': 0, 'True': 1}
    #label_map = {'Radiant': 1, 'Dire': 0}
    y_train = train['radiant_win'].astype(np.float32).values
    print(f'y_train:\n {y_train}')
    print('fitting time')
    classifier.fit(x=x_train, y=y_train)
    print('fit')

    # # Testing
    # x_test = test.drop(['radiant_win'], axis=1).astype(np.float32).values
    # y_test = test['radiant_win'].astype(np.float32).values
    #
    # y_out = list(classifier.predict(x=x_test))
    # print('test completed')
    # n = len(y_test)
    # out_hard = list(y['classes'] for y in y_out)
    # out_soft = list(y['probabilities'] for y in y_out)
    #
    # # SElf-note
    # # Soft predictions [dire win, radiant win]
    # print("Soft predictions:")
    # print(out_soft[:5])
    # print("Hard predictions:")
    # print(out_hard[:5])
    #
    # soft_zipped = zip(y_test, out_soft)
    # hard_zipped = list(zip(y_test, out_hard))
    #
    # num_correct = sum(1 for p in hard_zipped if p[0] == p[1])
    # print("Accuracy = %s" % (num_correct / n))
    #
    # test_ps = list(p[1][int(p[0])] for p in soft_zipped)
    # print("Probs of real label:")
    # print(test_ps[:5])
    # total_log_loss = sum(math.log(p) for p in test_ps)
    # print("Average log loss = %s" % (total_log_loss / n))
    # print()
    radiant = [20, 35, 14, 7, 32]
    dire = [17, 18, 26, 84, 96]
    predict(radiant, dire, classifier)
