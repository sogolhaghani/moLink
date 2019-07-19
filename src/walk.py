'''
Created on Mar 17, 2017

@author: root
'''
import numpy as np
import random

def randomWalk(G, n):
    walk = []
    source = random.choice(G.nodes())
    walk.append(source)
    count = 1
    while count < n:
        edges_out = G.out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        selected_edge = random.choice(edges_out)
#         label = G.get_edge_data(*selected_edge)
        target = selected_edge[1]
#         walk.append(label['label'])
        walk.append(target)
        count += 2
        source = target
    return walk

def windowWalk(param, source):
    if param['model'] == 1:
        return simple_window_walk(param, source)
    elif param['model'] == 2:
        return windowWalk1(param, source)
    elif param['model'] == 3:
        return windowWalk1(param, source)
    elif param['model'] == 4:
        return windowWalk4(param, source)
    elif param['model'] == 5:
        return windowWalk5(param, source)
    elif param['model'] == 6:
        return windowWalk6(param, source)
    elif param['model'] == 7:
        return windowWalk7(param, source)
    elif param['model'] == 8:
        return windowWalk8(param, source)
    elif param['model'] == 9:
        return windowWalk9(param, source)

def simple_window_walk(param, source):
    walk = []
    walk.append(source)
    count = 1
    while count < param['window']:
        edges_out = param['graph'].out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        selected_edge = random.choice(edges_out)
        target = selected_edge[1]
        walk.append(target)
        count += 1
        source = target
    return walk



def windowWalk1(param, source):
    G = param['graph']
    window = param['window']
    walk = []
    walk.append(source)
    count = 1
    t = 0
    while count < window:
        edges_out = G.out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk[::2]
        selected_edge = random.choice(edges_out)
        label = G[selected_edge[0]][selected_edge[1]]['label']
        target = selected_edge[1]
        pattern = [source, label, target]
        if ''.join(map(str, pattern)) in ''.join(map(str, walk)):
            t = t + 1
            if t > 2:
                return walk[::2]
            else:
                continue
        t = 0
        walk.append(label)
        walk.append(target)
        count += 1
        source = target
    return walk[::2]


def windowWalk2(param, source):
    G = param['graph']
    window = param['window']
    walk = []
    depth = 0
    depths = []
    walk.append(source)
    depths.append(depth)
    depth = depth + 1
    count = 1
    t = 0
    while count < window:
        edges_out = G.out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return compose_walk(walk, depths)
        weights = []
        for head, end in edges_out:
            pattern = [head, G[head][end]['label'], end]
            if ''.join(map(str, pattern)) in ''.join(map(str, walk)):
                weights.append(0)
            elif end not in walk:
                weight = len(G) / (len(G.out_edges(end)) + len(G.in_edges(end)))
                weights.append(weight)
            else:
                weights.append(20)
        selected_edge = select_edge(edges_out, weights)
        label = G[selected_edge[0]][selected_edge[1]]['label']
        end = selected_edge[1]
        pattern = [source, label, end]
        if ''.join(map(str, pattern)) in ''.join(map(str, walk)):
            t = t + 1
            if t > 2:
                return compose_walk(walk, depths)
            else:
                continue
        t = 0
        walk.append(label)
        depths.append(depth)
        walk.append(end)
        depths.append(depth)
        count += 1
        rand = random.uniform(0, 1)
        if rand > 0.5:
            source = end
            depth = depth + 1
    return compose_walk(walk, depths)

def windowWalk3(param, source):
    G = param['graph']
    window = param['window']
    walk = []
    depth = 0
    depths = []
    walk.append(source)
    depths.append(depth)
    depth = depth + 1
    count = 1
    t = 0
    while count < window:
        edges_out = G.out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return compose_walk(walk, depths)
        weights = []
        for head, end in edges_out:
            pattern = [head, G[head][end]['label'], end]
            if ''.join(map(str, pattern)) in ''.join(map(str, walk)):
                weights.append(0)
            elif end not in walk:
                weight = len(G) / (len(G.out_edges(end)) + len(G.in_edges(end)))
                weights.append(weight)
            else:
                weights.append(20)
        selected_edge = select_edge(edges_out, weights)
        label = G[selected_edge[0]][selected_edge[1]]['label']
        end = selected_edge[1]
        pattern = [source, label, end]
        if ''.join(map(str, pattern)) in ''.join(map(str, walk)):
            t = t + 1
            if t > 2:
                return compose_walk(walk, depths)
            else:
                continue
        t = 0
        walk.append(label)
        depths.append(depth)
        walk.append(end)
        depths.append(depth)
        count += 1
        depth = depth + 1
    return compose_walk(walk, depths)


def windowWalk4(param, source):
    G = param['graph']
    window = param['window']
    walk = []
    depth = 0
    depths = []
    walk.append(source)
    depths.append(depth)
    depth = depth + 1
    count = 1
    t = 0
    while count < window:
        edges_out = G.out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return compose_walk(walk, depths)
        weights = []
        for head, end in edges_out:
            pattern = [head, G[head][end]['label'], end]
            if ''.join(map(str, pattern)) in ''.join(map(str, walk)):
                weights.append(0)
            elif end not in walk:
                weight = len(G) / (len(G.out_edges(end)) + len(G.in_edges(end)))
                weights.append(weight)
            else:
                weights.append(20)
        selected_edge = select_edge(edges_out, weights)
        label = G[selected_edge[0]][selected_edge[1]]['label']
        end = selected_edge[1]
        pattern = [source, label, end]
        if ''.join(map(str, pattern)) in ''.join(map(str, walk)):
            t = t + 1
            if t > 2:
                return compose_walk(walk, depths)
            else:
                continue
        t = 0
        walk.append(label)
        depths.append(depth)
        walk.append(end)
        depths.append(depth)
        count += 1
    return compose_walk(walk, depths)

def compose_walk(walk, depths):
    a = zip(walk, depths)
    return a[::2]

def select_edge(edges_out, weights):
    if len(edges_out) == 1:
        return edges_out[0]
    summ = sum(weights)
    if summ == 0:
        return random.choice(edges_out)
    rand = random.randint(1, summ)
    summ = 0
    for index, weight in enumerate(weights):
        if rand > summ and rand <= summ + weight:
            return edges_out[index]
        else:
            summ = summ + weight
    return random.choice(edges_out)

####################################################3333
def windowWalk5(param, source):
    init_src = source
    walk = []
    walk.append(source)
    count = 1
    temp = 0
    while count < param['window']:
        edges_out = param['graph'].out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        selected_edge = random.choice(edges_out)
        if temp > 2 and len(edges_out) > 1 and param['graph'].has_edge(init_src, selected_edge[1]):
            temp = temp +1
            continue
        temp = 0
        target = selected_edge[1]
        walk.append(target)
        count += 1
        source = target
    return walk

def windowWalk6(param, source):
    walk = []
    walk.append(source)
    count = 1
    temp = 0
    while count < param['window']:
        edges_out = param['graph'].out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        selected_edge = random.choice(edges_out)
        if temp > 2 and len(edges_out) > 1 and selected_edge[1] not in walk:
            temp = temp +1
            continue
        temp = 0
        target = selected_edge[1]
        walk.append(target)
        count += 1
        source = target
    return walk

def windowWalk7(param, source):
    walk = []
    walk.append(source)
    count = 1
    temp = 0
    while count < param['window']:
        edges_out = param['graph'].out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        selected_edge = random.choice(edges_out)
        if temp > 2 and len(edges_out) > 1 and selected_edge[1] in walk:
            temp = temp +1
            continue
        temp = 0
        target = selected_edge[1]
        walk.append(target)
        count += 1
        source = target
    return walk

def windowWalk8(param, source):
    init_src = source
    walk = []
    walk.append(source)
    count = 1
    max_common_nei = 0
    selected_node = ''
    while count < param['window']:
        edges_out = param['graph'].out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        for head , end in edges_out:
            if len(common_out_neighbors(param['graph'], init_src, end))+ len(common_in_neighbors(param['graph'], init_src, end)) > max_common_nei:
                max_common_nei = len(common_out_neighbors(param['graph'], init_src, end)) + len(common_in_neighbors(param['graph'], init_src, end))
                selected_node = end
        if max_common_nei <= 0:
            selected_node = random.choice(edges_out)[1]
        target = selected_node
        walk.append(target)
        count += 1
        max_common_nei = -1
        selected_node = ''
        source = target
    return walk

def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))

def windowWalk9(param, source):
    walk = []
    walk.append(source)
    count = 1
    while count < param['window']:
        edges_out = param['graph'].out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        selected_edge = find_selected_edge(edges_out, param['graph'], param['iteration'])
        target = selected_edge[1]
        walk.append(target)
        count += 1
        source = target
    return walk

def find_selected_edge(edges_out , G , iter_num):
    if iter_num < 10:
        selected_edge = random.choice(edges_out)
        weights = G[selected_edge[0]][selected_edge[1]]['weight']
        G[selected_edge[0]][selected_edge[1]]['weight'] = weights + 1
        return selected_edge
    else:
        weights = np.zeros(len(edges_out))
        index = 0
        for head, end in edges_out:
            weights[index] = G[head][end]['weight']
        maxx = np.amax(weights)
        weights = np.subtract(maxx, weights)
        summ = np.sum(weights)
        rand = random.randint(1, summ)
        summ = 0
        for index, weight in enumerate(weights):
            if rand > summ and rand <= summ + weight:
                selected_edge = edges_out[index]
                edge_weight = G[selected_edge[0]][selected_edge[1]]['weight']
                G[selected_edge[0]][selected_edge[1]]['weight'] = edge_weight + 1
                return selected_edge
            else:
                summ = summ + weight
    return random.choice(edges_out)

