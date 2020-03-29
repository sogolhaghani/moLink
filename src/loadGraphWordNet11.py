'''
Created on Feb 15, 2017

@author: root
'''

import networkx as nx


def read_train():
    G = nx.DiGraph()
    _list = []
    file_edges = open('../data/wordnet/train.txt', 'r')
    for line in file_edges:
        tokens = line.split()
        if G.has_node(tokens[0]) is False:
            G.add_node(tokens[0])
        if G.has_node(tokens[2]) is False:
            G.add_node(tokens[2])
        G.add_edge(tokens[0], tokens[2], label=tokens[1], weight=0)
        _list.append(line.split())
    return G, _list

def read_test():
    f_test= open('../data/wordnet/test.txt', 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list    


def read_validation():
    f_test= open('../data/wordnet/dev.txt', 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list 
