'''
Created on Feb 15, 2017
Modified on May 31, 2017
@author: root
'''

import numpy as np
import evaluate as e1
import evaluate2 as e2
import activationFunction as af
import vocabulary as v
import walk as w
import TableForNegativeSamples as t


def init_parameter(param):
    param['vocab'] = v.Vocabulary(param['graph'])
    param['rel'] = param['vocab'].getRel()
    param['tnn0'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(len(param['vocab']), param['dim']))
    param['tnn1'] = np.zeros(shape=(len(param['vocab']), param['dim']))
    param['tnn2'] = np.random.uniform(low=-0.5 / len(param['rel']), high=0.5 / len(param['rel']) , size=(len(param['rel'])))
    param['table'] = t.TableForNegativeSamples(param['vocab'])
    return param

def refresh_parameter(param):
    param['nn0'] = param['tnn0']
    param['nn1'] = param['tnn1']
    param['nn2'] = param['tnn2']
    return param


def train(param):
    while param['iteration'] < param['iter_count']:
        param = link2vec1(param)
        if param['iteration'] > 20:
            validate(param)
        param['iteration'] += 1
        # print 'Iter : %s' % param['nn2']

def test(param):
    if param['evaluation_metric'] == 'auc':
        auc = e1.test(param['test'], param['nn0'], param['nn1'], param['nn2'], param['vocab'], param['rel'])
    elif param['evaluation_metric'] == 'mean_rank':
        auc = e2.test(param, test_file='test', is_filetered=False, is_bias=True)
    return auc

def validate(param):
    if param['evaluation_metric'] == 'auc':
        auc = e1.test(param['validation'], param['nn0'], param['nn1'], param['nn2'], param['vocab'], param['rel'])
    elif param['evaluation_metric'] == 'mean_rank':
        auc = e2.test(param, test_file='validation', is_filetered=False, is_bias=True)
    return auc


# Relation As Bias , simple window walk
def link2vec1(param):
    walks = list(param['graph'].edges_iter())
    for walk in walks:
        tokens = param['vocab'].indices(list(walk))
        for token_idx, token in enumerate(tokens):
            if (token_idx + 1) == len(tokens):
                continue
            label = param['graph'][walk[0]][walk[1]]['label']
            relation_index = param['rel'].index(label)
            source = param['vocab'].__getitem__(tokens[token_idx + 1]).word
            window_walk = w.windowWalk(param, source)
            contexts = param['vocab'].indices(window_walk)
            for context_idx, context in enumerate(contexts):
                head = param['vocab'].__getitem__(token).word
                tail = param['vocab'].__getitem__(context).word
                if param['graph'].has_edge(head, tail):
                    relation_index = param['rel'].index(param['graph'][head][tail]['label'])
                neu1e = np.zeros(param['dim'])
                alpha = float(param['alpha'])/ (context_idx +1)
                negative_samples = [(target, 0) for target in param['table'].sample(param['kns'])]
                classifiers = [(token, 1)] + negative_samples
                for target, label in classifiers:
                    target_word = param['vocab'].__getitem__(target).word
                    exist_edge = param['graph'].has_edge(tail, target_word)
                    if exist_edge and label == 0 and param['graph'][tail][target_word]['label'] == param['rel'][relation_index]:
                        continue
                    z_param = np.dot(param['nn0'][context], param['nn1'][target])
                    z_param += param['nn2'][relation_index]
                    probability = af.sigmoid(z_param)
                    entity_err = alpha * (label - probability)
                    rel_err = param['beta'] * (float(1)/(context_idx + 1) - probability)
                    neu1e += entity_err * param['nn1'][target]  # Error to backpropagate to nn0
                    param['nn1'][target] += entity_err * param['nn0'][context]  # Update nn1
                    param['nn2'][relation_index] += entity_err - rel_err
                param['nn0'][context] += neu1e
    return param
