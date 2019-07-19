'''
Created on Jan 28, 2017
Modified on May 31, 2017

@author: Sogol
'''
import time
import loadGraphWN18 as lgWN
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
    G, train_list =lgWN.__read_train()
    param['graph'] = G
    param['window'] = 1
    param['kns'] = 1
    param['alpha'] = 0.08  # 0.01
    param['beta'] = 0.08
    param['dim'] = 100
    param['iter_count'] = 20
    param['validation'] = lgWN.__read_validation()
    param['test'] = lgWN.__read_test()
    param['model'] = 1
    param['iteration'] = 0
    param['evaluation_metric'] = 'mean_rank'
    param['train'] = train_list
    return param



def tune_func(param, init, _step, _to, key):
    initial = init
    step = _step
    best = initial
    var = []
    auc_arr = []
    while initial < _to:
        param[key] = initial
        param = l2v.refresh_parameter(param)
        l2v.train(param)
        auc = l2v.validate(param)
        if len(auc_arr) > 0 and auc > auc_arr[-1]:
            best = initial
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
    param = l2v.init_parameter(param)
    alpha, alpha_auc, param = tune_func(param, 0.01, 0.005, 0.2, 'alpha')
    beta, beta_auc, param = tune_func(param, 0.001, 0.001, 0.2, 'beta')
    kns, kns_auc, param = tune_func(param, 1, 1, 20, 'k_negative_sampling')
    # window, window_auc, param = tune_func(param, 1, 1, 15, 'window')
    # iterr, iter_auc, param = tune_func(param, 10, 5, 150, 'iter_count')
    # dim, dim_auc, param = tune_func(param, 20, 10, 20, 'dim')
    model, model_auc, param = tune_func(param, 1, 1, 5, 'model')
    param = l2v.refresh_parameter(param)
    l2v.train(param)
    auc = l2v.test(param)
    print 'Test Auc : %s' % auc
    e.drawCurve(alpha, alpha_auc, 'Learning Rate', 'AUC', 'alpha', 'Parameter setting for Learning Rate')
    e.drawCurve(beta, beta_auc, 'BETA', 'AUC', 'beta', 'Parameter setting for beta')
    # e.drawCurve(window, window_auc, 'Window size', 'AUC', 'window', 'Parameter setting for Window size') 
    e.drawCurve(kns, kns_auc, 'k Negative Sample', 'AUC', 'k_negative_sample', 'Parameter setting for Negative sampling')
    # e.drawCurve(iterr, iter_auc, 'Iteration Count', 'AUC', 'iteration', 'Parameter setting for Iteration')
    # e.drawCurve(dim, dim_auc, 'Dimension', 'AUC', 'dimension', 'Parameter setting for dimension')
    e.drawCurve(model, model_auc, 'Function', 'AUC', 'model', 'random function')

if __name__ == "__main__":
    _main()
