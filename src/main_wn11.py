'''
Created on Jan 28, 2017
Modified on May 31, 2017

@author: Sogol
'''
import time
import loadGraphWordNet11 as lgWN
import link2vec as l2v
import evaluate as e


def timing(func):
    def wrap(*args):
        time1 = time.time()
        ret = func(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (func.func_name, (time2 - time1) * 1000.0)
        return ret
    return wrap


def init_param():
    param = {}
    G, train_list = lgWN.read_train()
    param['graph'] = G
    param['window'] = 9
    param['kns'] = 1
    param['alpha'] = 0.05  # 0.01
    param['beta'] = 0.0002
    param['dim'] = 100
    param['iter_count'] = 100
    param['validation'] = lgWN.read_validation()
    param['test'] = lgWN.read_test()
    param['model'] = 1
    param['iteration'] = 0
    param['evaluation_metric'] = 'auc'
    param['train'] = train_list
    return param



def tune_func(param, init, _step, _to, key):
    initial = init
    step = _step
    best = initial
    best_auc = 0
    var = []
    auc_arr = []
    while initial < _to:
        param[key] = initial
        param = l2v.refresh_parameter(param)
        l2v.train(param)
        auc = l2v.validate(param)
        if auc > best_auc:
            best = initial
            best_auc = auc
        var.append(initial)
        auc_arr.append(auc)
        initial += step
        param['iteration'] = 0
        print 'key : %s , value : %s , %s : %s' %(key, initial, param['evaluation_metric'], auc)
    print 'best %s : %s' % (key, best)
    param[key] = best
    return var, auc_arr, param

@timing
def _main():
    param = init_param()
    l2v.init_parameter(param)
    param = l2v.refresh_parameter(param)
    # alpha, alpha_auc, param = tune_func(param, 0.001, 0.005, 0.2, 'alpha')
    # beta, beta_auc, param = tune_func(param, 0.0001, 0.0001, 0.02, 'beta')
    # window, window_auc, param = tune_func(param, 9, 1, 15, 'window')
    # kns, kns_auc, param = tune_func(param, 1, 1, 20, 'k_negative_sampling')
    # iterr, iter_auc, param = tune_func(param, 10, 5, 150, 'iter_count')
    # dim, dim_auc, param = tune_func(param, 20, 10, 20, 'dim')
    # model, model_auc, param = tune_func(param, 1, 1, 5, 'model')
    # param = l2v.refresh_parameter(param)
    l2v.train(param)
    auc = l2v.test(param)
    print 'Test Auc : %s' % auc
    print 'nn2 : %s ' %param['nn2']
    # e.drawCurve(alpha, alpha_auc, 'Learning Rate', 'AUC', 'alpha', 'Parameter setting for Learning Rate')
    # e.drawCurve(beta, beta_auc, 'BETA', 'AUC', 'beta', 'Parameter setting for beta')
    # e.drawCurve(window, window_auc, 'Window size', 'AUC', 'window', 'Parameter setting for Window size') 
    # e.drawCurve(kns, kns_auc, 'k Negative Sample', 'AUC', 'k_negative_sample', 'Parameter setting for Negative sampling')
    # e.drawCurve(iterr, iter_auc, 'Iteration Count', 'AUC', 'iteration', 'Parameter setting for Iteration')
    # e.drawCurve(dim, dim_auc, 'Dimension', 'AUC', 'dimension', 'Parameter setting for dimension')
    # e.drawCurve(model, model_auc, 'Function', 'AUC', 'model', 'random function')

if __name__ == "__main__":
    _main()
