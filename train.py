# -*- coding: UTF-8 -*-
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import csv


def loadData():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    trainData = train.drop(['label'], axis=1).values.astype(dtype=np.int64)
    trainLabel = np.array(train['label'])

    testData = test.values
    print('load Data finish!!!')

    return trainData, trainLabel, testData


def saveResult(testLabel, fileName):
    header = ['ImageID', 'Label']
    with open(fileName, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        writer.writerow(header)
        for i, p in enumerate(testLabel):
            writer.writerow([str(i + 1), str(p)])

#随机森林分类
def RFClassify(trainData, trainLabel, testData):
    nbCF = RandomForestClassifier(n_estimators=256, warm_start=True)
    nbCF.fit(trainData, np.ravel(trainLabel))
    testLabel = nbCF.predict(testData)
    saveResult(testLabel, 'output/output5.csv')
    print('finish!!!')


RFClassify(*loadData())