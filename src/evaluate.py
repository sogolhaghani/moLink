
import numpy as np
from sklearn import metrics
import activationFunction as af

def accuracy(param, test_file='validation'):
    vocab = param['vocab']
    rel = param['rel']
    nn0 = param['nn0']
    nn1 = param['nn1']
    nn2 = param['nn2']
    test_list= param[test_file]
    classes = []
    predicted = []
    for  tokens in test_list:
        if  tokens[0] not in vocab or tokens[2] not in vocab:
            continue
        index_head = vocab.indices([tokens[0]])[0]
        index_rel = rel.index(tokens[1])
        index_tail = vocab.indices([tokens[2]])[0]
        class_tag = int(tokens[3])
        v_h = nn0[index_head, :]
        v_r = nn2[index_rel]
        v_t = nn1[index_tail, :]
        z_param = np.dot(v_h, v_t) + v_r
        v_1 = af.sigmoid(z_param)
        classes.append(class_tag)
        if v_1 > 0.5: #TODO 
            predicted.append(1)
        else:
            predicted.append(-1)    
    acc = metrics.accuracy_score(classes,predicted)
    return acc


def auc(param, test_file='validation'):
    vocab = param['vocab']
    rel = param['rel']
    nn0 = param['nn0']
    nn1 = param['nn1']
    nn2 = param['nn2']
    test_list= param[test_file]
    score = []
    classes = []
    for line_tokens in test_list:
        if  line_tokens[0] not in vocab or line_tokens[2] not in vocab:
            continue
        index_head = vocab.indices([line_tokens[0]])[0]
        index_rel = rel.index(line_tokens[1])
        index_tail = vocab.indices([line_tokens[2]])[0]
        class_tag = int(line_tokens[3])
        v_h = nn0[index_head, :]
        v_r = nn2[index_rel, :]
        v_t = nn1[index_tail, :]
        z_param = np.dot(v_h, v_t + v_r)
        v_1 = af.sigmoid(z_param)
        classes.append(class_tag)
        score.append(v_1)
    fpr, tpr, thresholds = metrics.roc_curve(classes, score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


# def meanRank(param, test_file='validation', is_filetered=True):
#     ranks = np.zeros(shape=(len(param[test_file]), 1))
#     vocab = param['vocab']
#     rel = param['rel']
#     for  index, tokens in enumerate(param[test_file]):
#         head = tokens[0]
#         relation = tokens[2]
#         tail = tokens[1]
#         if  head not in vocab or tail not in vocab or relation not in rel:
#             continue
#         head_index = vocab.indices([head])[0]
#         rel_index = rel.index(relation)
#         tail_index = vocab.indices([tail])[0]
#         corupted = []
#         if is_filetered:
#             corupted =__create_filtered_corupted(head_index, rel_index, vocab, param['train'], param['test'], param['validation'], tokens)
#         else:
#             corupted = __create_corupted(head_index, rel_index, vocab)
#         __calculate_scores_when_rel_is_bias(corupted, param['nn0'], param['nn1'], param['nn2'])
       
#         corupted = sorted(corupted, key=lambda a_entry: a_entry[3], reverse=True)
#         ranks[index] = __mean_rank(corupted, head_index, rel_index, tail_index)
#     mean_rank = np.mean(ranks)
#     return mean_rank
    