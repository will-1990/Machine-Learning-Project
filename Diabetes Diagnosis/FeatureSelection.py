from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV, RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def linear_svc_l1(X, Y):
    clf = svm.LinearSVC(C=0.002, penalty='l1', dual=False).fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print clf.coef_
    print np.mat(X).shape, X_new.shape
    print model.threshold_
    print model.get_support(indices=True)
    return X_new.tolist()


def linear_svc_l2(X, Y):
    clf = svm.LinearSVC().fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print clf.coef_
    print np.mat(X).shape, X_new.shape
    print model.threshold_
    print model.get_support(indices=True)
    return X_new.tolist()


def lr_l1(X, Y):
    clf = LogisticRegression(n_jobs=10, C=0.01, penalty='l1').fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print clf.coef_
    print np.mat(X).shape, X_new.shape
    print model.threshold_
    print model.get_support(indices=True)
    return X_new.tolist()


def lr_l2(X, Y):
    clf = LogisticRegression(n_jobs=10).fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print clf.coef_
    print np.mat(X).shape, X_new.shape
    print model.threshold_
    print model.get_support(indices=True)
    return X_new.tolist()


def XGB_Select(X, Y):
    clf = xgb.XGBClassifier(nthread=-1, n_estimators=500, max_depth=6, subsample=0.8, learning_rate=1,
                            colsample_bytree=1, gamma=1).fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print clf.feature_importances_
    print np.mat(X).shape, X_new.shape
    print model.threshold_
    print model.get_support(indices=True)
    return X_new.tolist()


def RF_Select(X, Y):
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=500, max_depth=7, max_features=0.6).fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print clf.feature_importances_
    print np.mat(X).shape, X_new.shape
    print model.threshold_
    print model.get_support(indices=True)
    return X_new.tolist()


def svm_FRECV(X, Y):
    estimator = svm.LinearSVC()
    selector = RFECV(estimator, step=10, cv=3, scoring='f1').fit(X, Y)
    X_new = selector.transform(X)
    print np.mat(X).shape, X_new.shape
    print selector.get_support(indices=True)
    plt.figure()
    plt.xlabel("Numbers of features selected")
    plt.ylabel("Cross Validation Score")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.show()
    return X_new.tolist()


def svm_FRE(X, Y):
    estimator = svm.LinearSVC()
    selector = RFE(estimator, step=10, n_features_to_select=20).fit(X, Y)
    X_new = selector.transform(X)
    print np.mat(X).shape, X_new.shape
    print selector.get_support(indices=True)
    return X_new.tolist()


def XGB_RFE(X, Y):
    estimator = xgb.XGBClassifier(nthread=10, n_estimators=500, max_depth=10, subsample=1, learning_rate=0.1,
                                  colsample_bytree=1, gamma=2)
    selector = RFE(estimator, step=10, n_features_to_select=20).fit(X, Y)
    X_new = selector.transform(X)
    print np.mat(X).shape, X_new.shape
    print selector.get_support(indices=True)
    return X_new.tolist()


def RF_RFE(X, Y):
    estimator = RandomForestClassifier(n_jobs=10, n_estimators=500, max_depth=7, max_features=0.6)
    selector = RFE(estimator, step=10, n_features_to_select=20).fit(X, Y)
    X_new = selector.transform(X)
    print np.mat(X).shape, X_new.shape
    print selector.get_support(indices=True)
    return X_new.tolist()


def univariate_chi(X, Y):
    selector = SelectKBest(chi2, k=20).fit(X, Y)
    X_new = selector.transform(X)
    print np.mat(X).shape, X_new.shape
    print selector.get_support(indices=True)
    return X_new.tolist()


def univariate_f(X, Y):
    selector = SelectKBest(f_classif, k=20).fit(X, Y)
    X_new = selector.transform(X)
    print np.mat(X).shape, X_new.shape
    print selector.get_support(indices=True)
    return X_new.tolist()