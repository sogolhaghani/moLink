'''
Created on Feb 1, 2017

@author: root
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import activationFunction as af
# from sklearn.metrics import average_precision_score

def test(test_list, nn0, nn1, nn2, vocab, rel):
    score = []
    classes = []
    predicted = []
    for  line_tokens in test_list:
        if  line_tokens[0] not in vocab or line_tokens[2] not in vocab:
            continue
        index_head = vocab.indices([line_tokens[0]])[0]
        index_rel = rel.index(line_tokens[1])
        index_tail = vocab.indices([line_tokens[2]])[0]
        class_tag = int(line_tokens[3])
        v_h = nn0[index_head, :]
        v_r = nn2[index_rel]
        v_t = nn1[index_tail, :]
        z_param = np.dot(v_h, v_t) + v_r
        v_1 = af.sigmoid(z_param)
        classes.append(class_tag)
        score.append(v_1)
        if v_1 > 0.5:
            predicted.append(1)
        else:
            predicted.append(-1)    
    # precision, recall, thresholds = metrics.precision_recall_curve(classes, score, pos_label=1)
    # auc_pr = metrics.auc(recall, precision)
    # auc_pr_a = average_precision_score(classes, score)
    fpr, tpr, thresholds = metrics.roc_curve(classes, score, pos_label=1)
    auc_roc = metrics.auc(fpr,tpr)
    acc = metrics.accuracy_score(classes,predicted)
    print 'Acc : %s' %acc
    return auc_roc


def test1(test_list, nn0, nn1, nn2, vocab, rel):
    score = []
    classes = []
    for line_tokens in test_list:
        if  line_tokens[0] not in vocab or line_tokens[2] not in vocab:
            continue
        index_head = vocab.indices([line_tokens[0]])[0]
        index_rel = rel.index(line_tokens[1])
        index_tail = vocab.indices([line_tokens[2]])[0]
        class_tag = int(line_tokens[3])
        v_h = nn0[index_head, :]
        v_r = nn2[index_rel, :]
        v_t = nn1[index_tail, :]
        z_param = np.dot(v_h, v_t + v_r)
        v_1 = af.sigmoid(z_param)
        classes.append(class_tag)
        score.append(v_1)
    fpr, tpr, thresholds = metrics.roc_curve(classes, score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def drawROCSingle(score , class_tag , p_label=1):
    fpr, tpr , thresholds = metrics.roc_curve(class_tag , score , pos_label=p_label) 
    plt.figure()
    lw = 2
    acu = 0
    try:
        acu = metrics.auc(fpr , tpr)
    except Exception as ex:
        print ex
        pass
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %s)' % acu)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic- WordNet')
    plt.legend(loc="lower right")
    plt.show()

def drawCurve(x , y , x_label , y_label , curve_label, title):
    plt.figure()
    plt.plot(x, y,'bo' , color='#B01F00', label=curve_label,linewidth=1.0, linestyle="-" , dash_joinstyle = 'bevel' , markeredgecolor='#B01F00')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
