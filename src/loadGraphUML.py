'''
Created on Feb 15, 2017

@author: root
'''




def read_graph():
    
    k_fold=
    
    file_relname = open('../data/uml/uml-relname.txt', 'r')
    rel_name =[]
    for line in file_relname:
        tokens = line.split()
        rel_name.append(tokens[0])
        
    print 'Relation name : %s' %rel_name    
        
    file_entityname = open('../data/uml/uml-entity-name.txt', 'r')
    entity_name = []
    for line in file_entityname:
        tokens = line.split()
        entity_name.append(tokens[0])
        
    print 'Entity name : %s' %entity_name               
    
    file_triple = open('../data/uml/pTriplets.txt', 'r')
    
    length = 0
    for line in file_triple:
        length += 1
    
    print 'triple len : %s' %length
     
    
            



if __name__ == "__main__":
    read_graph()