# encoding=utf-8

import soaplib
from soaplib.core.util.wsgi_wrapper import run_twisted
from soaplib.core.server import wsgi
from soaplib.core.service import DefinitionBase
from soaplib.core.service import soap
from soaplib.core.model.clazz import Array
from soaplib.core.model.clazz import ClassModel
from soaplib.core.model.primitive import Double, Integer
import isDiabetic

class PredictResults(ClassModel):
    __namespace__ = "MyModel"
    result_1 = Double
    result_2 = Double


class MyService(DefinitionBase):  # this is a web service

    @soap(Integer, Array(Double), _returns=PredictResults)
    def predict(self, i, X):
        print 'sample ', i, X
        print 'length of X is:', len(X)
        predict_results = PredictResults()
        predict_results.result_1, predict_results.result_2 = isDiabetic.predict(X)
        return predict_results


if __name__ == '__main__':  # 发布服务
    try:
        print '服务已经开启'
        from wsgiref.simple_server import make_server

        soap_application = soaplib.core.Application([MyService], 'tns')
        wsgi_application = wsgi.Application(soap_application)
        server = make_server('localhost', 8888, wsgi_application)
        server.serve_forever()

    except ImportError:
        print 'error'

