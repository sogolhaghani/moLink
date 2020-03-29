from sklearn import metrics
import matplotlib.pyplot as plt

def drawROCSingle(score , class_tag , p_label=1):
    fpr, tpr , thresholds = metrics.roc_curve(class_tag , score , pos_label=p_label) 
    plt.figure()
    lw = 2
    acu = 0
    try:
        acu = metrics.auc(fpr , tpr)
    except Exception as ex:
        print(ex)
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