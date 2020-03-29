'''
Created on Mar 17, 2017

@author: Sogol Haghani
'''
import numpy as np
import random

def windowWalk(param, source):
    walk = []
    walk.append(source)
    count = 1
    while count < param['window']:
        edges_out = param['graph'].out_edges(source)
        if edges_out is None or len(edges_out) == 0:
            return walk
        selected_edge = random.choice(list(edges_out))
        target = selected_edge[1]
        walk.append(target)
        count += 1
        source = target
    return walk