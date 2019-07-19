import os
import cPickle

import random
import networkx as nx


def k_fold_cross_validation():
# Number of folds
    k_fold = 10
    datapath = '../data/kinships_pkl/'
    assert datapath is not None
    if 'data' not in os.listdir('../'):
            os.mkdir('../data')
    for dataset in ['kinships']:
        file = open('/home/sogol/wspy/TestML1/data/kinships_pkl/kinships.pkl')
        # f = open(datapath + dataset + '.pkl')
        dictdata = cPickle.load(file)
        tensordata = dictdata['tensor']
        file.close()
        # List non-zeros
        list_positive = []
        # List zeros
        list_negative = []
        # Fill the lists
        for i in range(tensordata.shape[0]):
            for j in range(tensordata.shape[1]):
                for k in range(tensordata.shape[2]):
                    if tensordata[i, j, k] == 0:
                        list_negative += [(i, k, j, 0)]
                    elif tensordata[i, j, k] == 1:
                        list_positive += [(i, k, j, 1)]
        len_n = len(list_negative)
        len_p = len(list_positive)
        test = []
        validation = []
        train_list = []
        #Test
        for j in range(len_p/k_fold):
            index = random.randint(0, len(list_positive)-1)
            head = list_positive[index][0]
            relation = list_positive[index][1]
            tail = list_positive[index][2]
            class_tag = list_positive[index][3]
            test.append((head, relation, tail, class_tag))
            del list_positive[index]
        for j in range(len_n/k_fold):
            index = random.randint(0, len(list_negative)-1)
            head = list_negative[index][0]
            relation = list_negative[index][1]
            tail = list_negative[index][2]
            class_tag = list_negative[index][3]
            test.append((head, relation, tail, class_tag))
            del list_negative[index]
        #Validation
        for j in range(len_p/k_fold):
            index = random.randint(0, len(list_positive)-1)
            head = list_positive[index][0]
            relation = list_positive[index][1]
            tail = list_positive[index][2]
            class_tag = list_positive[index][3]
            validation.append((head, relation, tail, class_tag))
            del list_positive[index]
        for j in range(len_n/k_fold):
            head = list_negative[index][0]
            relation = list_negative[index][1]
            tail = list_negative[index][2]
            class_tag = list_negative[index][3]
            validation.append((head, relation, tail, class_tag))
            del list_negative[index]
        graph = nx.DiGraph()
        for tokens in list_positive:
            if graph.has_node(tokens[0]) is False:
                graph.add_node(tokens[0])
            if graph.has_node(tokens[2]) is False:
                graph.add_node(tokens[2])
            graph.add_edge(tokens[0], tokens[2], label=tokens[1], weight=0)
            train_list.append((tokens[0], tokens[1], tokens[2], tokens[3]))
        train_n = []
        for tokens in list_negative:
            # train_list.append((tokens[0], tokens[1], tokens[2], tokens[3]))
            train_n.append((tokens[0], tokens[1], tokens[2], tokens[3]))
        return graph, validation, test, train_list, train_n
#         return G, train_list, train_n
 
    