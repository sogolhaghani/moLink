'''
Created on Feb 1, 2017
Modified on April 21, 2017
Modified on June 2, 2017
@author: root
'''
import numpy as np
import activationFunction as af

def test(param, test_file='validation', is_bias=True, is_filetered=False):
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
        if is_bias:
            __calculate_scores_when_rel_is_bias(corupted, param['nn0'], param['nn1'], param['nn2'])
        else:
            __calculate_scores_when_rel_is_vector(corupted, param['nn0'], param['nn1'], param['nn2'])
        corupted = sorted(corupted, key=lambda a_entry: a_entry[3], reverse=True)
        ranks[index] = __mean_rank(corupted, head_index, rel_index, tail_index)
        hits[index] = __hits_top(corupted, head_index, rel_index, tail_index, top_hits)
    mean_rank = np.mean(ranks)
    hits = np.average(hits)
    print 'Mean Rank: %s' %mean_rank
    print 'Hits %s : %s' %(top_hits, hits)
    return hits

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
    corupted = np.zeros(shape=(len(vocab), 4))
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
    print 'size : %s ' %size
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

def __calculate_scores_when_rel_is_vector(corupted, nn0, nn1, nn2):
    for index in range(len(corupted[:, 0])):
        head_index = corupted[index][0]
        rel_index = corupted[index][1]
        tail_index = corupted[index][2]
        v_h = nn0[head_index, :]
        v_r = nn2[rel_index, :]
        v_t = nn1[tail_index, :]
        z_param = np.dot(v_h, v_t + v_r)
        corupted[index][3] = af.sigmoid(z_param)
    return corupted
