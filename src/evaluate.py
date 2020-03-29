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


def test(param, test_file='validation', is_filetered=False):
    hits = np.zeros(shape=(len(param[test_file]), 1))
    ranks = np.zeros(shape=(len(param[test_file]), 1))
    top_hits = 10
    vocab = param['vocab']
    rel = param['rel']
    for  index, line_tokens in enumerate(param[test_file]):
        head = line_tokens[0]
        relation = line_tokens[2]
        tail = line_tokens[1]
        if  head not in vocab or tail not in vocab or relation not in rel:
            continue
        head_index = vocab.indices([head])[0]
        rel_index = rel.index(relation)
        tail_index = vocab.indices([tail])[0]
        corupted = []
        if is_filetered:
            corupted =__create_filtered_corupted(head_index, rel_index, vocab, param['train'], param['test'], param['validation'], line_tokens)
        else:
            corupted = __create_corupted(head_index, rel_index, vocab)
        __calculate_scores_when_rel_is_bias(corupted, param['nn0'], param['nn1'], param['nn2'])
        corupted = sorted(corupted, key=lambda a_entry: a_entry[3], reverse=True)
        ranks[index] = __mean_rank(corupted, head_index, rel_index, tail_index)
        hits[index] = __hits_top(corupted, head_index, rel_index, tail_index, top_hits)
    mean_rank = np.mean(ranks)
    hits = np.average(hits)
    mean_rank = np.mean(ranks)
    hits = np.average(hits)
    return hits, top_hits, mean_rank

def __mean_rank(corupted, head_index, rel_index, tail_index):
    for index in range(1000):
        c_head_index = corupted[index][0]
        c_rel_index = corupted[index][1]
        c_tail_index = corupted[index][2]
        if c_head_index == head_index and c_rel_index == rel_index and c_tail_index == tail_index:
            return index + 1
    return 1000

def __hits_top(corupted, head_index, rel_index, tail_index, top_hits):
    for index in range(top_hits):
        c_head_index = corupted[index][0]
        c_rel_index = corupted[index][1]
        c_tail_index = corupted[index][2]
        if c_head_index == head_index and c_rel_index == rel_index and c_tail_index == tail_index:
            return 1
    return 0

def __create_corupted(head_index, rel_index, vocab):
    corupted = np.zeros(shape=(len(vocab), 4), dtype=np.int32)
    for index, word in enumerate(vocab):
        corupted[index][0] = head_index
        corupted[index][1] = rel_index
        corupted[index][2] = vocab.indices([word.word])[0]
    return corupted

def __create_filtered_corupted(head_index, rel_index, vocab, train, test, validation, triple):
    size = 0
    corupted = np.zeros(shape=(len(vocab), 4))
    index = 0
    for word in  vocab:
        c_triple = [triple[0], word.word, triple[2]]
        if c_triple == triple or (c_triple not in train and c_triple not in test and c_triple not in validation):
            size += 1
            corupted[index][0] = head_index
            corupted[index][1] = rel_index
            corupted[index][2] = vocab.indices([word.word])[0]
            index += 1
    return corupted[0: size]

def __calculate_scores_when_rel_is_bias(corupted, nn0, nn1, nn2):
    for index in range(len(corupted[:, 0])):
        head_index = corupted[index][0]
        rel_index = corupted[index][1]
        tail_index = corupted[index][2]
        v_h = nn0[head_index, :]
        v_r = nn2[rel_index]
        v_t = nn1[tail_index, :]
        z_param = np.dot(v_h, v_t) + v_r
        corupted[index][3] = af.sigmoid(z_param)
    return corupted
    