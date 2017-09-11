# encoding=utf-8

# import SqlConnector
import datetime
import Classifiers
# import FeatureSelection
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import Sampling
import csv

# 记录日志,不过输出有点混乱,估计跟跨文件有关,后续再弄
logfile = open('log.txt', 'a')

'''
def test():
    ms = SqlConnector.MSSQL(host='.', user='sa', password='qwe123!@#', database='dibao', port=1433)
    sql = 'select top 100 * from HR_HealthCheck'
    sql1 = 'select top 100 * from HR_Person'
    resultList = ms.execQury(sql1)

    for item in resultList:
        print item[0], encode(item[4])


def encode(code):
    return code.encode('unicode_escape').decode('string_escape').decode('gbk')


# 用SQL处理好之后，直接导入样本
def newLoadData():
    ms = SqlConnector.MSSQL(host='.', user='sa', password='qwe123!@#', database='dibao', port=1433)
    database0 = "new_samples"  # 缺失值用所有样本均值填充
    database1 = "samples_1"  # 缺失值用填充5个临近点均值
    database2 = "samples_2"  # 缺失值用线性插值法填充
    database3 = "samples_3"  # 增加了新的特征值,加入每人最高血糖值,最低血糖值，还有平均血糖值，缺失值用所有样本均值填充
    database4 = "samples_4_"  # 增加更多的特征值
    database5 = "samples_5"  # 删除了最大血糖值为空的负样本，总样本大约43W
    sql = "select top 10000 * from " + database4 + " where diabetes='1' union select top 100000 * from " + database4 + " where diabetes='0' "
    resultList = ms.execQury(sql)

    return resultList
'''

def loadCSV():
    csvfile = open('D:\Diabetes\sample\samples_new.csv', 'r')
    reader = csv.reader(csvfile)
    xlist = []
    ylist = []
    for sample in reader:
        if reader.line_num == 1:
            continue

        features = []
        for i in range(6):
            features.append(int(sample[i + 1]))  # sex,年龄和家族史
        features.extend(sample[7:-1])  # 体检信息
        label = int(sample[-1])

        xlist.append(features)
        ylist.append(label)

    xlist = preprocessing.minmax_scale(xlist)

    return xlist, ylist


def featureSelect(xlist, ylist):
    # xlist = FeatureSelection.univariate_chi(xlist,ylist)
    # xlist = FeatureSelection.univariate_f(xlist, ylist)
    # xlist = FeatureSelection.XGB_RFE(xlist, ylist)
    # xlist = FeatureSelection.RF_RFE(xlist, ylist)
    # xlist = FeatureSelection.XGB_Select(xlist, ylist)
    # xlist = FeatureSelection.RF_Select(xlist, ylist)
    # xlist = FeatureSelection.linear_svc_l2(xlist, ylist)
    # xlist = FeatureSelection.linear_svc_l1(xlist, ylist)
    # xlist = FeatureSelection.lr_l1(xlist, ylist)
    # xlist = FeatureSelection.lr_l2(xlist, ylist)

    return xlist


'''
# 早期特征工程,只有21个特征,而且顺序有点混乱
def featureSelect(resultList):
    xlist = []
    ylist = []
    for sample in resultList:
        features = []
        for i in range(12):
            features.append(sample[i + 1])  # 最近体检记录
        for i in range(5):
            features.append(int(sample[i + 13]))  # sex 家族病史
        features.append(sample[-5])  # 加入年龄
        label = int(sample[-4])
        features.extend(sample[-3:])  # 最高血糖值,最低血糖值，还有平均血糖值
        xlist.append(features)
        ylist.append(label)

    xlist = preprocessing.minmax_scale(xlist).tolist()
    # xlist = FeatureSelection.linear_svc_l2(xlist, ylist)
    # xlist = FeatureSelection.linear_svc_l1(xlist, ylist)
    # xlist = FeatureSelection.svm_FRECV(xlist, ylist)
    return xlist, ylist


# 后面的特征工程,总共60个特征
def featureSelect_1(resultList):
    xlist = []
    ylist = []
    for sample in resultList:
        features = []
        for i in range(6):
            features.append(int(sample[i + 1]))  # sex,年龄和家族史
        features.extend(sample[7:-1])  # 体检信息
        label = int(sample[-1])

        xlist.append(features)
        ylist.append(label)

    xlist = preprocessing.minmax_scale(xlist)
    xlist = FeatureSelection.linear_svc_l2(xlist, ylist)
    # xlist = FeatureSelection.linear_svc_l1(xlist, ylist)
    # xlist = FeatureSelection.svm_FRECV(xlist, ylist)
    # xlist = FeatureSelection.svm_FRE(xlist, ylist)
    # xlist = FeatureSelection.univariate(xlist,ylist)
    return xlist, ylist
'''


def main():
    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    start_time = datetime.datetime.now()
    print 'operation start time : %s     author : will' % start_time
    print '----------------step 1: import samples-----------------'
    start_time = datetime.datetime.now()
    # queryList = newLoadData()
    X, Y = loadCSV()
    print 'the total samples:%d' % len(Y)
    end_time = datetime.datetime.now()
    print 'the time for step 1:%s seconds' % (end_time - start_time).seconds

    print '----------------step 2:feature selection-------------------'
    start_time = datetime.datetime.now()
    # X, Y = featureSelect(queryList)
    X = featureSelect(X, Y)
    end_time = datetime.datetime.now()
    print 'the time for step 2:%s seconds' % (end_time - start_time).seconds

    print '----------------step 3:train and test set------------------'

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=3)

    '''
    train_x = X[:40000]
    train_x.extend(X[50000:90000])
    test_x = X[40000:50000]
    test_x.extend(X[90000:])
    train_y = Y[:40000]
    train_y.extend(Y[50000:90000])
    test_y = Y[40000:50000]
    test_y.extend(Y[90000:])
    '''

    positive_train = train_y.count(1)
    positive_test = test_y.count(1)
    negative_train = train_y.count(0)
    negative_test = test_y.count(0)
    print "training set  :%8d   testing set   :%8d" % (len(train_y), len(test_y))
    print "positive train:%8d   negative train:%8d" % (positive_train, negative_train)
    print "positive test :%8d   negative test :%8d" % (positive_test, negative_test)

    print '----------------step 4:model training---------------'

    print '##################### original ###########################'
    pre1=Classifiers.classify(train_x, train_y, test_x, test_y, False)
    '''
    print '##################### cost sensitive #####################'

    Classifiers.classify(train_x, train_y, test_x, test_y, True)

    print '##################### random under sample ################'
    start_time = datetime.datetime.now()
    new_train_x, new_train_y = Sampling.randomUnderSample(train_x, train_y)
    end_time = datetime.datetime.now()
    print 'the time for sampling: %s seconds' % (end_time - start_time).seconds
    Classifiers.classify(new_train_x, new_train_y, test_x, test_y, False)
    '''
    print '##################### easy ensemble ######################'
    pre2=Sampling.easyEnsemble(train_x, train_y, test_x, test_y)

    '''
    print '##################### random subspace ######################'
    Sampling.RSEnsemble(train_x, train_y, test_x, test_y)

    end_time = datetime.datetime.now()
    print 'operation end time : %s    author : will' % end_time
    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    '''

    two_model_matrix(pre1, pre2, test_y)
    '''
    print '##################### ENN sample #########################'
    start_time = datetime.datetime.now()
    new_train_x, new_train_y = Sampling.ENN(train_x, train_y)
    end_time = datetime.datetime.now()
    print 'the time for sampling: %s seconds' % (end_time - start_time).seconds
    Classifiers.classify(new_train_x, new_train_y, test_x, test_y, False)


    print '##################### cluster center #####################'
    start_time = datetime.datetime.now()
    new_train_x, new_train_y = Sampling.ClusterCen(train_x, train_y)
    end_time = datetime.datetime.now()
    print 'the time: %s seconds' % (end_time - start_time).seconds
    classify(new_train_x, new_train_y, test_x, test_y, False)

    print '##################### smote #############################'
    start_time = datetime.datetime.now()
    new_train_x, new_train_y = Sampling.Smote(train_x, train_y)
    end_time = datetime.datetime.now()
    print 'the time: %s seconds' % (end_time - start_time).seconds
    Classifiers.classify(new_train_x, new_train_y, test_x, test_y, False)
    '''

def two_model_matrix(pre1, pre2, true):
    count = [0] * 8

    for i in range(len(true)):
        count[4 * pre1[i] + 2 * pre2[i] + true[i]] += 1

    print count


if __name__ == '__main__':
    # test()
    main()
