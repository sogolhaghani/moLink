import networkx as nx

def train(src , hIndex=0 , rIndex=1, tIndex=2):
    G = nx.DiGraph()
    _list = []
    file_edges = open(src, 'r')
    for line in file_edges:
        tokens = line.split()
        if G.has_node(tokens[hIndex]) is False:
            G.add_node(tokens[hIndex])
        if G.has_node(tokens[tIndex]) is False:
            G.add_node(tokens[tIndex])
        G.add_edge(tokens[hIndex], tokens[tIndex], label=tokens[rIndex], weight=0)
        _list.append(line.split())
    return G, _list

def test(src):
    f_test= open(src, 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list       