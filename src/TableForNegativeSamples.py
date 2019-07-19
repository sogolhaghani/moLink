'''
Created on Mar 17, 2017

@author: root
'''
import math
import random

import numpy as np


class TableForNegativeSamples:
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constants

        table_size = 1e5
        table = np.zeros(table_size, dtype=np.int32)

        p = 0  # Cumulative probability
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]
    
    
 
def get_k_negativeSamples(negativesamples, k , head,vocab):
    temp = []
    for sample in negativesamples:
        if(sample[0] == head):
            temp.append(sample[1])
            
    result = []        
    for i in range(k):
        if(len(temp)==0):
            return result
        rand_index = random.randint(0,len(temp)-1)    
        result.append(( vocab.indices([temp[rand_index]])[0],0))     
        del temp[rand_index]    
    return result