# encoding = utf-8

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import math
from numpy import *
import datetime
import xgboost as xgb
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

scorer = make_scorer(fbeta_score, beta=3)


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

    auc = roc_auc_score(true, pre_pro)
    # fpr, tpr, thresholds = roc_curve(true, pre_pro, pos_label=1)

    print "the sensitivity :%.4f    the specificity :%.4f" % (sensitivity, specificity)
    print "the posPrecision:%.4f    the negPrecision:%.4f" % (pos_precision, neg_precision)
    print "the f-beta is   :%.4f    the G-mean is   :%.4f" % (f_beta, G_mean)
    print "the auc is      :%.4f" % auc

    '''
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''

class_wei={0:1,1:70}

def LR(train_x, train_y, test_x, test_y, cost_sensitive):
    lr = LogisticRegression(n_jobs=-1, C=0.1)
    if cost_sensitive:
        lr.set_params(class_weight=class_wei)
    # grid = GridSearchCV(lr, param_grid={'C': [0.1, 1, 10]}, scoring=scorer, cv=5)
    # grid.fit(train_x, train_y)
    # print grid.grid_scores_
    # print grid.best_score_, grid.best_params_
    # pre = grid.predict(test_x)
    lr.fit(train_x, train_y)
    pre = lr.predict_proba(test_x)
    pred = [x[1] for x in pre]
    Performance(pred, test_y, 0.5)
    # print lr.coef_


def DT(train_x, train_y, test_x, test_y, cost_sensitive):
    dt = tree.DecisionTreeClassifier(max_depth=12)
    if cost_sensitive:
        dt.set_params(class_weight=class_wei)
        # grid = GridSearchCV(dt, param_grid={'max_depth': [11, 12, 13],
        # 'max_features': [None, 0.6, 'sqrt', 'log2']},
        # scoring=scorer, cv=5)
    # grid.fit(train_x, train_y)
    # print grid.grid_scores_
    # print grid.best_score_, grid.best_params_
    # pre = grid.predict(test_x)
    dt.fit(train_x, train_y)
    pre = dt.predict_proba(test_x)
    pred = [x[1] for x in pre]
    Performance(pred, test_y, 0.5)
    # print dt.feature_importances_


def Linear_SVC(train_x, train_y, test_x, test_y, cost_sensitive):
    linear_svc = svm.LinearSVC()
    if cost_sensitive:
        linear_svc.set_params(class_weight=class_wei)
    # grid = GridSearchCV(linear_svc, param_grid={'C': [0.01, 0.1, 1, 10]}, scoring='precision', cv=3)
    # grid.fit(train_x, train_y)
    # print grid.grid_scores_
    # print grid.best_score_, grid.best_params_
    #pre = grid.predict_proba(test_x)
    linear_svc.fit(train_x, train_y)
    pre = linear_svc.predict(test_x)
    Performance(pre, test_y, 0.5)
    # return pre
    # print linear_svc.coef_


def RBF_SVC(train_x, train_y, test_x, test_y, cost_sensitive):
    rbf_svc = svm.SVC()
    rbf_svc.fit(train_x, train_y)
    pre = rbf_svc.predict_proba(test_x)
    if cost_sensitive:
        rbf_svc.set_params(class_weight=class_wei)
    # grid = GridSearchCV(rbf_svc, param_grid={'C': [0.1, 1, 10], 'gamma': ['auto', 0.1, 1.0, 10]}, scoring='precision', cv=3)
    # grid.fit(train_x, train_y)
    # print grid.grid_scores_
    # print grid.best_score_, grid.best_params_
    # pre = grid.predict(test_x)
    Performance(pre, test_y, 0.5)
    # return pre


def randomForest(train_x, train_y, test_x, test_y, cost_sensitive):
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, max_depth=7, max_features=0.6)
    # rf = RandomForestClassifier(n_jobs=-1, n_estimators=500)
    if cost_sensitive:
        rf.set_params(class_weight=class_wei)

        # grid = GridSearchCV(rf, param_grid={'max_depth': [6, 7, 8, 9, 10],
        # 'max_features': ['sqrt', 'log2', 0.6, None]},
        # scoring='precision', cv=5)
    # grid.fit(train_x, train_y)
    # print grid.grid_scores_
    # print grid.best_score_, grid.best_params_
    # pre = grid.predict(test_x)

    rf.fit(train_x, train_y)
    pre = rf.predict_proba(test_x)
    pred = [x[1] for x in pre]
    Performance(pred, test_y, 0.5)
    # print rf.feature_importances_


def xgboost(train_x, train_y, test_x, test_y):
    xg = xgb.XGBClassifier(nthread=-1, n_estimators=500, max_depth=10, subsample=1, learning_rate=0.1,
                           colsample_bytree=1, gamma=2)
    # xg = xgb.XGBClassifier(nthread=22, n_estimators=200)
    '''grid = GridSearchCV(xg, param_grid={'max_depth': [6, 10, 15],
                                        'learning_rate': [0.1, 1],
                                        'gamma': [1, 2, 5],
                                        'subsample': [0.8, 1.0],
                                        'colsample_bytree': [0.8, 1.0]}, scoring='precision', cv=3)
    grid.fit(train_x, train_y)
    print grid.grid_scores_
    print grid.best_score_, grid.best_params_
    pre = grid.predict(test_x)
    '''
    xg.fit(train_x, train_y)
    pre = xg.predict_proba(test_x)
    pred = [x[1] for x in pre]

    Performance(pred, test_y, 0.5)
    indexes = repeat(0, array(pred).shape[0])
    indexes[array(pred) >= 0.5] = 1
    return indexes


def adaboost(train_x, train_y, test_x, test_y):
    ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)
    '''
    grid = GridSearchCV(ada, param_grid={'learning_rate': [0.1, 1]},
                        scoring=scorer, cv=5)
    grid.fit(train_x, train_y)
    print grid.grid_scores_
    print grid.best_score_, grid.best_params_
    pre = grid.predict(test_x)'''
    ada.fit(train_x, train_y)
    pre = ada.predict_proba(test_x)
    pred = [x[1] for x in pre]
    Performance(pred, test_y, 0.5)


def classify(train_x, train_y, test_x, test_y, cost_sensitive):
    if cost_sensitive:
        print class_wei
    '''
    print '******************** logistic regression ***************'
    start_time = datetime.datetime.now()
    LR(train_x, train_y, test_x, test_y, cost_sensitive)
    end_time = datetime.datetime.now()
    print 'the time for LR: %s seconds' % (end_time - start_time).seconds

    print '******************** decision tree *********************'
    start_time = datetime.datetime.now()
    DT(train_x, train_y, test_x, test_y, cost_sensitive)
    end_time = datetime.datetime.now()
    print 'the time for DT: %s seconds' % (end_time - start_time).seconds

    print '********************* linear svc ***********************'
    start_time = datetime.datetime.now()
    Linear_SVC(train_x, train_y, test_x, test_y, cost_sensitive)
    end_time = datetime.datetime.now()
    print 'the time for Linear_SVC: %s seconds' % (end_time - start_time).seconds

    print '******************** adaboost **************************'
    start_time = datetime.datetime.now()
    adaboost(train_x, train_y, test_x, test_y)
    end_time = datetime.datetime.now()
    print 'the time for adaboost: %s seconds' % (end_time - start_time).seconds

    print '******************** random forest *********************'
    start_time = datetime.datetime.now()
    randomForest(train_x, train_y, test_x, test_y, cost_sensitive)
    end_time = datetime.datetime.now()
    print 'the time for random forest: %s seconds' % (end_time - start_time).seconds
    '''
    print '******************** xgboost ***************************'
    start_time = datetime.datetime.now()
    pre = xgboost(train_x, train_y, test_x, test_y)
    end_time = datetime.datetime.now()
    print 'the time for xgboost: %s seconds' % (end_time - start_time).seconds
    return pre
    '''
    print '********************* rbf svc *************************'
    start_time = datetime.datetime.now()
    Classifiers.RBF_SVC(train_x, train_y, test_x, test_y)
    end_time = datetime.datetime.now()
    print 'the time for RBF_SVC: %s seconds' % (end_time - start_time).seconds
    '''
