import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#import tensorflow as tf
import xgboost as xgb
from MinHeap import MinHeap

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
        print(np.mean(cross_val_score(rf, splittedData[0], splittedData[2], cv=10)))

    else:
        rf.fit(splittedData[0], splittedData[1])
    return rf

def scikitMLP(splittedData, training = False):
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

    if training is True:
        mlp.fit(splittedData[0], splittedData[2])
        predictions = mlp.predict(splittedData[1])

        print(confusion_matrix(splittedData[3], predictions))
        print(classification_report(splittedData[3], predictions))
        print(accuracy_score(splittedData[3], predictions))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(splittedData[3], predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print(roc_auc)
        print(np.mean(cross_val_score(mlp, splittedData[0], splittedData[2], cv=10)))

    else:
        mlp.fit(splittedData[0], splittedData[1])
    return mlp

def scikitTest(splittedData, training = False):
    test = QuadraticDiscriminantAnalysis()

    if training is True:
        test.fit(splittedData[0], splittedData[2])
        predictions = test.predict(splittedData[1])

        print(confusion_matrix(splittedData[3], predictions))
        print(classification_report(splittedData[3], predictions))
        print(accuracy_score(splittedData[3], predictions))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(splittedData[3], predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print(roc_auc)
        print(np.mean(cross_val_score(test, splittedData[0], splittedData[2], cv=10)))

    else:
        test.fit(splittedData[0], splittedData[1])
    return test

# def tensorFlowRForest(splittedData, training = False):
#     params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
#         num_classes=2, num_features=len(splittedData[0]), max_nodes=1000).fill()
#
#     classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
#         params, model_dir="./tensorflowmodel")
#
#     if training is True:
#         classifier.fit(x=splittedData[0], y=splittedData[2])
#         y_out = list(classifier.predict(x=splittedData[1]))
#         n = len(splittedData[3])
#         out_soft = list(y['classes'] for y in y_out)
#         out_hard = list(y['probabilities'] for y in y_out)
#
#         print(confusion_matrix(splittedData[3], out_soft))
#         print(classification_report(splittedData[3], out_soft))
#         print(accuracy_score(splittedData[3], out_soft))
#
#         false_positive_rate, true_positive_rate, thresholds = roc_curve(splittedData[3], out_soft)
#         roc_auc = auc(false_positive_rate, true_positive_rate)
#         print(roc_auc)
#
#     else:
#         classifier.fit(x=splittedData[0], y=splittedData[1])
#     return classifier

def xgBoost(splittedData, training = False):
    #train_dmatrix = xgb.DMatrix(data=splittedData[0], label=splittedData[2])
    if training is True:
        xg_model = xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.2) \
            .fit(splittedData[0], splittedData[2])
        predictions = xg_model.predict(splittedData[1])
        softpredictions = xg_model.predict_proba(splittedData[1])
        print(softpredictions)
        print(confusion_matrix(splittedData[3], predictions))
        print(classification_report(splittedData[3], predictions))
        print(accuracy_score(splittedData[3], predictions))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(splittedData[3], predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print(roc_auc)
        print(np.mean(cross_val_score(xg_model, splittedData[0], splittedData[2], cv=10)))

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

def predictLastPick(radiant, dire, classifiers):
    uniqueColList = [24, 115, 116, 117, 118, 122, 123, 124, 125, 126, 127, 128]
    heroesScreened = []
    data = []
    if len(radiant) == 4:
        counter = 0
    else:
        counter = 129
    for i in range(129):
        if counter + 1 not in radiant and counter - 129 + 1 not in dire and counter + 1 not in uniqueColList and counter - 129 + 1 not in uniqueColList:
            newRow = [0] * 258
            for i in radiant:
                newRow[int(i)-1] = 1
            for i in dire:
                newRow[int(i)+129-1] = 1

            for i in range(len(uniqueColList)):
                newRow[uniqueColList[i] - 1] = 2
                newRow[uniqueColList[i] + 129 - 1] = 2

            newRow[counter] = 1
            while 2 in newRow:
                newRow.remove(2)
            data.append(newRow)
            id = counter
            if counter > 128:
                id -= 128
            heroesScreened.append(id+1)
        counter += 1

    data = np.array(data)
    scikitprob_out = classifiers[0].predict_proba(data)
    nnmlpprob_out = classifiers[1].predict_proba(data)
    #y_out = list(classifiers[1].predict(x=data))
    #tensorout_soft = list(y['probabilities'] for y in y_out)
    xgbprob_out = classifiers[2].predict_proba(data)
    testprob_out = classifiers[3].predict_proba(data)

    heroesHeap = MinHeap(5)
    for i in range(len(xgbprob_out)):
        probability = (scikitprob_out[i] + nnmlpprob_out[i] + xgbprob_out[i] + testprob_out[i]) / 4
        if len(heroesHeap) < 5:
            if len(radiant) == 4:
                heroesHeap.push([heroesScreened[i], probability[1]])
            else:
                heroesHeap.push([heroesScreened[i], probability[0]])
        else:
            if len(radiant) == 4:
                if probability[1] > heroesHeap.array[1][1]:
                    heroesHeap.pop()
                    heroesHeap.push([heroesScreened[i], probability[1]])
            else:
                if probability[0] > heroesHeap.array[1][1]:
                    heroesHeap.pop()
                    heroesHeap.push([heroesScreened[i], probability[0]])

    heroProb = []
    for i in range(len(heroesHeap)):
        heroProb.append(heroesHeap.pop())

    return heroProb[::-1]


if __name__ == "__main__":
    matchData = readData("testMatches_noBool.csv")
    cleanedData = cleanData(matchData)
    #split = splitData(cleanedData[0], cleanedData[1])
    #np.random.seed(9999)
    #scikitRForest(split, True)
    # tf.set_random_seed(9999)
    # tensorFlowRForest(split)
    # np.random.seed(9999)
    # xgBoost(split)
    #scikitMLP(split, True)
    #scikitTest(split, True)
    #scikitRForest(split, True)
    #xgBoost(split, True)

    radiant = [12, 27, 33, 45, 53]
    dire = [26, 17, 78, 119, 106]
    # rforest = scikitRForest(cleanedData)
    # print(predictResult(radiant, dire, rforest))
    #
    # nnmlp = scikitMLP(cleanedData)
    # print(predictResult(radiant, dire, nnmlp))
    #
    # scikittest = scikitTest(cleanedData)
    # print(predictResult(radiant, dire, scikittest))
    #
    # xgboost = xgBoost(cleanedData)
    # print(predictResult(radiant, dire, xgboost))

    #tforest = tensorFlowRForest(cleanedData)
    #print(predictResult(radiant, dire, tforest, True))

    #xgbmodel = xgBoost(cleanedData)
    #print(predictLastPick(radiant, dire, [xgbmodel]))
    #print(predictResult(radiant, dire, xgbmodel))



