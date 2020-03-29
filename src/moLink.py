'''
Created on Feb 15, 2017
Modified on May 31, 2017
@author: root
'''

import numpy as np
import evaluate as e
import activationFunction as af
import walk as w


def train(param):
    while param['iteration'] < param['iter_count']:
        param = moLink(param) 
        if param['iteration'] % 5 == 0:
            evaluation(param)
        param['iteration'] += 1 
         

def evaluation(param, test_file='validation'):
    if param['evaluation_metric'] == 'auc':
        acc = e.accuracy( param , test_file)
        print('Iteration %s, Accuracy: %s' %(param['iteration'] ,acc))

    elif param['evaluation_metric'] == 'mean_rank':
        hits, top_hits, mean_rank = e.test(param, test_file='validation', is_filetered=False)
        print('Iteration %s, mean_rank: %s, top_hits: %s, hits %s' %(param['iteration'] ,mean_rank, top_hits, hits))


def moLink(param):
    walks = list(param['graph'].edges)
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
