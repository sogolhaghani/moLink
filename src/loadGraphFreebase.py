'''
Created on Feb 15, 2017

@author: root
'''

import networkx as nx


def __read_train():
    graph = nx.DiGraph()
    file_edges = open('../data/freebase15k/train.txt', 'r')
    for line in file_edges:
        tokens = line.split()
        if graph.has_node(tokens[0]) is False:
            graph.add_node(tokens[0])
        if graph.has_node(tokens[2]) is False:
            graph.add_node(tokens[2])
        graph.add_edge(tokens[0], tokens[2], label=tokens[1])
    return graph

def __read_test():
    f_test = open('../data/freebase15k/test.txt', 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list


def __read_validation():
    f_test = open('../data/freebase15k/valid.txt', 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list
