'''
Created on Jan 28, 2017
Modified on May 31, 2017

@author: Sogol
'''
import time
import numpy as np
import loadData as lg
import moLink as l2v
import evaluate as e
import TableForNegativeSamples as t
import vocabulary as v


def timing(func):
    def wrap(*args):
        time1 = time.time()
        ret = func(*args)
        time2 = time.time()
        print( '%s function took %0.3f ms' %(func.func_name, (time2 - time1) * 1000.0))
        return ret
    return wrap


def init_paramWN11():
    param = {}
    G, train_list = lg.train('./data/wordnet/train.txt')
    param['graph'] = G
    param['train'] = train_list
    param['validation'] = lg.test('./data/wordnet/dev.txt')
    param['test'] = lg.test('./data/wordnet/test.txt')
    param['window'] = 9
    param['kns'] = 1
    param['alpha'] = 0.05  # 0.01
    param['beta'] = 0.0002
    param['dim'] = 100
    param['iter_count'] = 20
    param['iteration'] = 0
    param['evaluation_metric'] = 'auc'
    param['vocab'] = v.Vocabulary(param['graph'])
    param['rel'] = param['vocab'].getRel()
    param['nn0'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(len(param['vocab']), param['dim']))
    param['nn1'] = np.zeros(shape=(len(param['vocab']), param['dim']))
    param['nn2'] = np.random.uniform(low=-0.5 / len(param['rel']), high=0.5 / len(param['rel']) , size=(len(param['rel'])))
    param['table'] = t.TableForNegativeSamples(param['vocab'])

    return param


@timing
def _main():
    param = init_paramWN11()
    l2v.train(param)
    print('Test Result') 
    l2v.evaluation(param, test_file='test')

if __name__ == "__main__":
    _main()
