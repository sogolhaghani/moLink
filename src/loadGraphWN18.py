'''
Created on May 31, 2017

@author: Sogol
'''

import networkx as nx


def __read_train():
    graph = nx.DiGraph()
    file_edges = open('/home/sogol/Desktop/ws_w2v_bias/data/WN18/train.txt', 'r')
    test_list = []
    for line in file_edges:
        tokens = line.split()
        if graph.has_node(tokens[0]) is False:
            graph.add_node(tokens[0])
        if graph.has_node(tokens[1]) is False:
            graph.add_node(tokens[1])
        graph.add_edge(tokens[0], tokens[1], label=tokens[2])
        test_list.append(line.split())
    return graph, test_list

def __read_test():
    f_test = open('/home/sogol/Desktop/ws_w2v_bias/data/WN18/test.txt', 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list


def __read_validation():
    f_test = open('/home/sogol/Desktop/ws_w2v_bias/data/WN18/valid.txt', 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list
