from suds.client import Client
from DiabeticPredictInterface.isDiabetic import load_data

client = Client('http://localhost:8888/SOAP/?wsdl')
ser = client.service

X, y, _ = load_data('D:\Diabetes\sample\samples_test.csv')

negative_count = 0
for i in range(len(y)):
    if y[i] == 1 or negative_count < 20:
        negative_count += 1
        print 'sample %d:' % i
        sample = client.factory.create("predict")
        sample.X = X[i]
        result = ser.predict(i, sample)
        if result.result_1 > result.result_2:
            print 'error'
        else:
            print result, y[i]

