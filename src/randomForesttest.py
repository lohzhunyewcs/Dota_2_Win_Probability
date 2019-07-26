import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import xgboost as xgb


def readData(filename):
    readData = pd.read_csv(filename)
    return readData

def cleanData(data):
    data = data.drop('match_id', axis=1)
    labels = np.array(data['radiant_win'])
    data = data.drop('radiant_win', axis=1)

    resultList = list(data.columns)

    uniqueColList = []
    for columns in resultList:
        if len(data[columns].unique()) == 1:
            uniqueColList.append(columns)

    for i in range(len(uniqueColList)):
        data = data.drop(uniqueColList[i], axis=1)

    data = np.array(data)
    return [data, labels]

def splitData(data, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(data, labels, test_size = 0.25, random_state = 99)
    return [train_features, test_features, train_labels, test_labels]

def scikitRForest(splittedData, training = False):
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 99)

    if training is True:
        rf.fit(splittedData[0], splittedData[2])
        predictions = rf.predict(splittedData[1])

        print(confusion_matrix(splittedData[3], predictions))
        print(classification_report(splittedData[3], predictions))
        print(accuracy_score(splittedData[3], predictions))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(splittedData[3], predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print(roc_auc)
        # print(np.mean(cross_val_score(rf, splittedData[0], splittedData[2], cv=10)))

    else:
        rf.fit(splittedData[0], splittedData[1])
    return rf

def tensorFlowRForest(splittedData, training = False):
    params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_classes=2, num_features=len(splittedData[0]), max_nodes=1000).fill()

    classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
        params, model_dir="./")

    if training is True:
        classifier.fit(x=splittedData[0], y=splittedData[2])
        y_out = list(classifier.predict(x=splittedData[1]))
        n = len(splittedData[3])
        out_soft = list(y['classes'] for y in y_out)
        out_hard = list(y['probabilities'] for y in y_out)

        print(confusion_matrix(splittedData[3], out_soft))
        print(classification_report(splittedData[3], out_soft))
        print(accuracy_score(splittedData[3], out_soft))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(splittedData[3], out_soft)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print(roc_auc)

    else:
        classifier.fit(x=splittedData[0], y=splittedData[1])
    return classifier

def xgBoost(splittedData, training = False):
    #train_dmatrix = xgb.DMatrix(data=splittedData[0], label=splittedData[2])
    if training is True:
        xg_model = xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.2) \
            .fit(splittedData[0], splittedData[2])
        predictions = xg_model.predict(splittedData[1])

        print(confusion_matrix(splittedData[3], predictions))
        print(classification_report(splittedData[3], predictions))
        print(accuracy_score(splittedData[3], predictions))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(splittedData[3], predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print(roc_auc)

    else:
        xg_model = xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.2) \
            .fit(splittedData[0], splittedData[1])
    return xg_model
    #print(np.mean(cross_val_score(rf, splittedData[0], splittedData[2], cv=10)))

def predictResult(radiant, dire, classifier, tensor = False):
    uniqueColList = [24, 115, 116, 117, 118, 122, 123, 124, 125, 126, 127, 128]
    data = np.zeros(258, dtype=int)
    for i in radiant:
        data[int(i)-1] = 1
    for i in dire:
        data[int(i) + 129 - 1] = 1

    for i in range(len(uniqueColList)):
        data[uniqueColList[i] - 1] = 2
        data[uniqueColList[i] + 129 - 1] = 2

    data = data[data != 2]
    data = np.array([data]).astype(np.float32)

    if tensor is False:
        prob_out = classifier.predict_proba(data)
        class_out = classifier.predict(data)
        #print(prob_out) # Soft predictions [dire win, radiant win]
        #print(class_out)
        return prob_out[0][1]
    else:
        y_out = list(classifier.predict(x=data))
        out_hard = list(y['classes'] for y in y_out)
        out_soft = list(y['probabilities'] for y in y_out)

        return out_soft[0][1]


if __name__ == "__main__":
    matchData = readData("testMatches_noBool.csv")
    cleanedData = cleanData(matchData)
    #split = splitData(cleanedData[0], cleanedData[1])
    #np.random.seed(9999)
    #scikitRForest(split)
    # tf.set_random_seed(9999)
    # tensorFlowRForest(split)
    # np.random.seed(9999)
    # xgBoost(split)

    radiant = [20, 35, 14, 7, 32]
    dire = [17, 18, 26, 84, 96]
    #rforest = scikitRForest(cleanedData)
    #predictResult(radiant, dire, rforest)

    #tforest = tensorFlowRForest(cleanedData)
    #print(predictResult(radiant, dire, tforest, True))

    xgbmodel = xgBoost(cleanedData)
    print(predictResult(radiant, dire, xgbmodel))



