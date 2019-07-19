'''
Created on Feb 15, 2017

@author: root
'''


class Word:
    def __init__(self, word , count):
        self.word = word
        self.count = count

class Vocabulary:
    def __init__(self, nx_G):
        self.words = []
        self.word_map = {}
        self.rel = []
        self.build_words(nx_G)
        self.build_rel(nx_G)

#     @timing
    def build_words(self, nx_G):
        words = []
        word_map = {}
        for token in nx_G.nodes():
            word_map[token] = len(words)
            words.append(Word(token , nx_G.degree(token)))
        print ("\rVocabulary built: %d" % len(words))
        self.words = words
        self.word_map = word_map  # Mapping from each token to its index in vocab
    
    def   build_rel(self, nx_G):
        rel = []
        for (u,v,d) in nx_G.edges(data='label'):
            if(d not in  rel):
                rel.append(d)
        self.rel = rel
    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __contains__(self, key):
        return key in self.word_map

    def indices(self, tokens):
        return [self.word_map[token] for token in tokens]
    
    def indices1(self, tokens):
        return [(self.word_map[token[0]] , token[1]) for token in tokens]
    
    
#[(target, 0) for target in param['table'].sample(param['k_negative_sampling'])]    
    def getRel(self):
        return self.rel