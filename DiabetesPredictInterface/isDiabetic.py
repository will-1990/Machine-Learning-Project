# encoding=utf-8
'''
notes:
author: will
date: 2017-4-21
'''

import csv
import os
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
from datetime import datetime
from xEnsemble import XEnsemble
from sklearn.metrics import precision_recall_fscore_support
import math

train_data_path = '.\samples\samples_new.csv'  # 重新预处理训练数据集
test_data_path = '.\samples\samples_test_1.csv'
file_path = './save_model'
file_max_min_avg = file_path + '/max_min_avg.pkl'
file_model_1 = file_path + '/model_1.pkl'
file_model_2 = file_path + '/model_2.pkl'

feature_num = 60


def load_data(filePath, need_maxmin):
    file = open(filePath, 'r')
    reader = csv.reader(file)
    X = []
    y = []
    max_min_avg = []
    for sample in reader:
        if reader.line_num == 1:
            continue
        if sample[1] != '0' and sample[1] != '1':
            continue

        features = []
        for i in range(60):
            features.append(float(sample[i + 1]))  # sex, age, family history, health check info
        label = int(sample[-1])

        X.append(features)
        y.append(label)

    if need_maxmin:
        samples = np.array(X).shape[0]
        features = np.array(X).shape[1]
        for i in range(features):
            temp = []
            max_i = max(np.array(X)[:, i])
            min_i = min(np.array(X)[:, i])
            avg_i = sum(np.array(X)[:, i]) / samples
            temp.append(max_i)
            temp.append(min_i)
            temp.append(avg_i)
            max_min_avg.append(temp)

    return X, y, max_min_avg


def train_model():
    print '------- load data -------'
    start_time = datetime.now()
    X, y, max_min_avg = load_data(train_data_path, True)
    X = preprocessing.minmax_scale(X)
    end_time = datetime.now()
    print 'the time for loading data: %s seconds' % (end_time - start_time).seconds

    print '-------train model 1-------'
    start_time = datetime.now()
    xgb = XGBClassifier(nthread=-1, n_estimators=500, max_depth=10, subsample=1, learning_rate=0.1,
                        colsample_bytree=1, gamma=2)
    model_1 = xgb.fit(X, y)
    end_time = datetime.now()
    print 'the time for training model 1: %s seconds' % (end_time - start_time).seconds

    print '-------train model 2-------'
    start_time = datetime.now()
    base_clf = XGBClassifier(nthread=-1, n_estimators=500, max_depth=10, subsample=1, learning_rate=0.1,
                             colsample_bytree=1, gamma=2)
    xen = XEnsemble(base_estimator=base_clf, n_subsets=20)
    model_2 = xen.fit(X, y)
    end_time = datetime.now()
    print 'the time for training model 2: %s seconds' % (end_time - start_time).seconds

    print '-------save model 1 and model 2--------'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    joblib.dump(max_min_avg, file_max_min_avg)
    joblib.dump(model_1, file_model_1)
    joblib.dump(model_2, file_model_2)

    return model_1, model_2, max_min_avg


def get_model():
    print '****************** get model **********************'
    exist_model_1 = os.path.exists(file_model_1) and os.path.getsize(file_model_1)
    exist_model_2 = os.path.exists(file_model_2) and os.path.getsize(file_model_2)
    exist_max_min_avg = os.path.exists(file_max_min_avg) and os.path.getsize(file_max_min_avg)
    if exist_model_1 and exist_model_2 and exist_max_min_avg:
        print '---------- load model 1 and model 2 -----------'
        model_1 = joblib.load(file_model_1)
        model_2 = joblib.load(file_model_2)
        max_min_avg = joblib.load(file_max_min_avg)
    else:
        print '---------- train model 1 and model 2 -----------'
        print 'maybe need long time, please wait...'
        model_1, model_2, max_min_avg = train_model()

    return model_1, model_2, max_min_avg


def predictOne(X):
    model_1, model_2, max_min_avg = get_model()

    if len(X) != feature_num:
        raise ValueError('error! feature number is not 60!')
    # judge whether input sample is outlier , and standardlize input sample
    X = checkOne(X, max_min_avg)

    result_1 = model_1.predict_proba(X)[0][1]
    result_2 = model_2.predict_proba(X)[0][1]

    return round(result_1, 3), round(result_2, 3)


def checkOne(X, max_min_avg):
    for i in range(len(X)):
        if X[i] > 1.2 * max_min_avg[i][0]:
            raise ValueError(
                'error! feature %d:%f is 1.2 larger than max record:%d' % (i, X[i], max_min_avg[i][0]))
        elif max_min_avg[i][0] <= X[i] < 1.2 * max_min_avg[i][0]:
            X[i] = 1.0
        elif max_min_avg[i][1] <= X[i] < max_min_avg[i][0]:
            X[i] = (X[i] - max_min_avg[i][1]) / (max_min_avg[i][0] - max_min_avg[i][1])
        elif 0.8 * max_min_avg[i][1] <= X[i] < max_min_avg[i][1]:
            X[i] = 0
        elif X[i] < 0.8 * max_min_avg[i][1] and X[i] != -1:
            raise ValueError(
                'error! feature %d:%f is 0.8 smaller than min record:%d' % (i, X[i], max_min_avg[i][1]))
        elif X[i] == -1:
            X[i] = (max_min_avg[i][2] - max_min_avg[i][1]) / (max_min_avg[i][0] - max_min_avg[i][1])  # let missing value be avg value

    return X


def check(sample, max_min_avg):
    is_normal = True
    for i in range(len(sample)):
        if sample[i] == -1:
            sample[i] = (max_min_avg[i][2] - max_min_avg[i][1]) / (max_min_avg[i][0] - max_min_avg[i][1])
        elif sample[i] > max_min_avg[i][0] or sample[i] < max_min_avg[i][1]:
            is_normal = False
            break
        else:
            sample[i] = (sample[i] - max_min_avg[i][1]) / (max_min_avg[i][0] - max_min_avg[i][1])

    return sample, is_normal


def predictAll(X):
    samples, features = np.array(X).shape
    print 'samples:%d, features:%d' % (samples, features)

    model_1, model_2, max_min_avg = get_model()
    outliers = 0
    X_new = []
    normal_indx = []
    for i in range(len(X)):
        new_sample, is_normal = check(X[i], max_min_avg)
        if is_normal:
            X_new.append(new_sample)
            normal_indx.append(i)
        else:
            outliers += 1

    pred_1 = model_1.predict_proba(X_new)[:, 1]
    pred_2 = model_2.predict_proba(X_new)[:, 1]

    print 'the number of outliers is: ', outliers
    return normal_indx, pred_1, pred_2


def Performance(pre_pro, true, thres):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    predict = []
    for x in pre_pro:
        if x >= thres:
            predict.append(1)
        else:
            predict.append(0)

    for i in range(len(predict)):
        if predict[i] == 1 and true[i] == 1:
            TP += 1
        elif predict[i] == 1 and true[i] == 0:
            FP += 1
        elif predict[i] == 0 and true[i] == 1:
            FN += 1
        elif predict[i] == 0 and true[i] == 0:
            TN += 1
    sensitivity = float(TP) / (TP + FN)
    specificity = float(TN) / (TN + FP)
    pos_precision = float(TP) / (TP + FP)
    neg_precision = float(TN) / (TN + FN)
    f_beta = 10 * sensitivity * pos_precision / (sensitivity + 9 * pos_precision)
    G_mean = math.sqrt(sensitivity * specificity)

    print "the sensitivity :%.4f    the specificity :%.4f" % (sensitivity, specificity)
    print "the posPrecision:%.4f    the negPrecision:%.4f" % (pos_precision, neg_precision)
    print "the f-beta is   :%.4f    the G-mean is   :%.4f" % (f_beta, G_mean)


def init():
    get_model()


def main_1():
    X, y, _ = load_data(train_data_path, False)
    X = preprocessing.minmax_scale(X)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=3)

    model_1, model_2, _ = get_model()

    score_1 = model_1.score(test_x, test_y)
    score_2 = model_2.score(test_x, test_y)
    print 'the accuracy for medel 1 and 2 are:', score_1, score_2
    pred_1 = model_1.predict(test_x)
    pred_2 = model_2.predict(test_x)
    print precision_recall_fscore_support(test_y, pred_1, labels=[1])
    print precision_recall_fscore_support(test_y, pred_2, labels=[1])


def main_2():
    X, y, _ = load_data('.\samples\samples_test.csv', False)

    for i in range(50):
        print 'sample %d:' % i
        print predictOne(X[i]), y[i]

def main_3():
    X, y, _ = load_data(test_data_path, False)

    normal_indx, pred_1, pred_2 = predictAll(X)
    test_y = np.array(y)[normal_indx]
    test_y = test_y.tolist()

    Performance(pred_1, test_y, 0.5)
    Performance(pred_2, test_y, 0.5)


if __name__ == '__main__':
    # init()
    # main_1()
    # main_2()
    main_3()
