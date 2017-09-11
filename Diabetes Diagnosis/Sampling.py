# encoding=utf-8

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsemble
from numpy import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import Classifiers
import datetime

def randomUnderSample(X, Y):
    sampler = RandomUnderSampler(replacement_max=False)
    X_new, Y_new = sampler.fit_sample(X, Y)
    print 'the original shape:%s   the resample shape:%s' % (mat(Y).shape, Y_new.shape)
    return X_new, Y_new


def ENN(X, Y):
    sampler = EditedNearestNeighbours(n_jobs=20)
    X_new, Y_new = sampler.fit_sample(X, Y)
    print 'the original shape:%s   the resample shape:%s' % (mat(Y).shape, Y_new.shape)
    return X_new, Y_new


def ClusterCen(X, Y):
    sampler = ClusterCentroids(n_jobs=12)
    X_new, Y_new = sampler.fit_sample(X, Y)
    print 'the original shape:%s   the resample shape:%s' % (mat(Y).shape, Y_new.shape)
    return X_new, Y_new

def Smote(X, Y):
    sampler = SMOTE(n_jobs=20, kind='borderline1')
    X_new, Y_new = sampler.fit_sample(X, Y)
    print 'the original shape:%s   the resample shape:%s' % (mat(Y).shape, Y_new.shape)
    return X_new, Y_new


def random_subspace(n_features, n_subsets, subset_features):
    feature_range = range(n_features)
    feature_subsets = []
    for i in range(n_subsets):
        random_state = random
        feature_ind = random_state.choice(feature_range, subset_features, replace=False)
        feature_ind = sorted(feature_ind)
        feature_subsets.append(feature_ind)

    return feature_subsets


def easyEnsemble(train_x, train_y, test_x, test_y):
    start_time = datetime.datetime.now()
    n_subsets = 20
    ee = EasyEnsemble(n_subsets=n_subsets, replacement_max=False, replacement_min=False)
    X_new, Y_new = ee.fit_sample(train_x, train_y)

    '''
    X_en, Y_en = ee.fit_sample(train_x, train_y)
    X_new = []
    Y_new = []
    smote = SMOTE(n_jobs=12, kind='svm')
    for i in range(n_subsets):
        X_sm, Y_sm = smote.fit_sample(X_en[i], Y_en[i])
        X_new.append(X_sm)
        Y_new.append(Y_sm)
    '''

    print 'the original shape:%s   the resample shape:%s' % (mat(train_y).shape, mat(Y_new).shape)
    end_time = datetime.datetime.now()
    print 'the time for sampling: %s seconds' % (end_time - start_time).seconds

    return ensemble_classify(X_new, Y_new, test_x, test_y, None)


def RSEnsemble(train_x, train_y, test_x, test_y):
    start_time = datetime.datetime.now()
    n_subsets = 35
    ee = EasyEnsemble(n_subsets=n_subsets, replacement_max=False, replacement_min=False)
    X_new, Y_new = ee.fit_sample(train_x, train_y)

    subset_features = 60
    feature_subsets = random_subspace(n_features=array(train_x).shape[1], n_subsets=n_subsets,
                                      subset_features=subset_features)

    print 'the original shape:%s   the resample shape:%s' % (mat(train_y).shape, mat(Y_new).shape)
    print 'the feature subset:%d' % subset_features
    end_time = datetime.datetime.now()
    print 'the time for sampling: %s seconds' % (end_time - start_time).seconds

    ensemble_classify(X_new, Y_new, test_x, test_y, feature_subsets)


def ensemble_classify(X_new, Y_new, test_x, test_y, feature_subsets):
    '''
    print  '******************** logistic regression ***************'
    start_time = datetime.datetime.now()
    predict_pro(X_new, Y_new, test_x, test_y, feature_subsets, LogisticRegression(n_jobs=-1, C=0.1))
    end_time = datetime.datetime.now()
    print 'the time for LR: %s seconds' % (end_time - start_time).seconds

    print '******************** decision tree *********************'
    start_time = datetime.datetime.now()
    predict_pro(X_new, Y_new, test_x, test_y, feature_subsets, tree.DecisionTreeClassifier(max_depth=11))
    end_time = datetime.datetime.now()
    print 'the time for DT: %s seconds' % (end_time - start_time).seconds

    print '********************* linear svc ***********************'
    start_time = datetime.datetime.now()
    predict_vote(X_new, Y_new, test_x, test_y, feature_subsets, svm.LinearSVC(C=0.1))
    end_time = datetime.datetime.now()
    print 'the time for Linear_SVC: %s seconds' % (end_time - start_time).seconds

    print '******************** adaboost ***************************'
    start_time = datetime.datetime.now()
    predict_pro(X_new, Y_new, test_x, test_y, feature_subsets, AdaBoostClassifier(n_estimators=200, learning_rate=0.1))
    end_time = datetime.datetime.now()
    print 'the time for vote adaboost: %s seconds' % (end_time - start_time).seconds

    print '******************** random forest *********************'
    start_time = datetime.datetime.now()
    predict_pro(X_new, Y_new, test_x, test_y, feature_subsets, RandomForestClassifier(n_jobs=-1, n_estimators=500, max_depth=7, max_features=0.6))
    end_time = datetime.datetime.now()
    print 'the time for random forest: %s seconds' % (end_time - start_time).seconds
    '''
    print '******************** xgboost ***************************'
    start_time = datetime.datetime.now()
    pre = predict_pro(X_new, Y_new, test_x, test_y, feature_subsets, xgb.XGBClassifier(nthread=-1, n_estimators=500, max_depth=10, subsample=1, learning_rate=0.1, colsample_bytree=1, gamma=2))
    end_time = datetime.datetime.now()
    print 'the time for xgboost: %s seconds' % (end_time - start_time).seconds
    return pre

# 将每个基模型的弱分类器线性求和来判断正负例
def predict_pro(X_new, Y_new, test_x, test_y, feature_subsets, classifier):
    pre_list = []
    n_subsets = len(Y_new)
    for i in range(n_subsets):
        x_train = X_new[i]
        x_test = test_x
        if feature_subsets is not None:
            x_train = X_new[i][:, feature_subsets[i]]
            x_test = test_x[:, feature_subsets[i]]
        clf = classifier.fit(x_train, Y_new[i])
        pre_result = clf.predict_proba(x_test)
        pre_pos = [x[1] for x in pre_result]
        pre_list.append(pre_pos)

    pre_mat = mat(pre_list)
    temp = mat(ones((1, n_subsets)), dtype=int)
    pre_pro = temp * pre_mat / n_subsets
    pre_pro_list = pre_pro.tolist()[0]

    Classifiers.Performance(pre_pro_list, test_y, 0.5)
    predict = []
    for x in pre_pro_list:
        if x >= 0.5:
            predict.append(1)
        else:
            predict.append(0)
    return predict


# 用投票法来判断正负例，累加基模型的正例个数
def predict_vote(X_new, Y_new, test_x, test_y, feature_subsets, classifier):
    pre_list = []
    n_subsets = len(Y_new)
    for i in range(n_subsets):
        x_train = X_new[i]
        x_test = test_x
        if feature_subsets is not None:
            x_train = X_new[i][:, feature_subsets[i]]
            x_test = test_x[:, feature_subsets[i]]
        clf = classifier.fit(x_train, Y_new[i])
        pre_result = clf.predict(x_test)
        pre_list.append(pre_result)

    pre_mat = mat(pre_list)
    temp = mat(ones((1, n_subsets)), dtype=int)
    pre_vote = temp * pre_mat / n_subsets
    pre_vote_list = pre_vote.tolist()[0]

    Classifiers.Performance(pre_vote_list, test_y, 0.5)

