import pandas as pd
import matplotlib.pyplot as plt
from random import randint
def tsne(X,y, k_fold,ar, p_time):
    """

    :param x:
    :param y:
    :param savep_path:
    :return:
    """


    from sklearn.manifold import TSNE
    from cycler import cycler

    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # Se = (green 'yellow' 'brown' 'orange' 'pink'  'black' 'gray'  'red' 'magenta'  'blue')

    
   #  for i in range(10):
   #      colors.append('#%06X' % randint(0, 0xFFFFFF))
   # colors = (colors)
    # print(plt.cm.Set2(1))
    plt.figure(figsize=(10,10))
    for i in range(X_norm.shape[0]):
        # print(y[i])
        if y[i]==1:
            color1='red'
        elif y[i]==2:
            color1='green'
        elif y[i]==3:
            color1='brown' 
        elif y[i]==4:
            color1='pink'  
        elif y[i]==5:
            color1='black'
        elif y[i]==6:
            color1='blue' 
        elif y[i]==7:
            color1='orange'
        elif y[i]==8:
            color1='green'
        elif y[i]==9:
            color1='magenta'
        else:
            color1='yellow'
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=color1,fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])

    #plt.show()

    plt.savefig('{}-fold_tSNE'+str(ar)+'.png'.format(k_fold))
    plt.close()

    save_path_excel = '{}-fold_tSNE'+str(ar)+'.xls'.format(k_fold)
    pd.set_option('display.max_columns', None)
    each_fold = pd.DataFrame(index=[str(i) for i in range(len(X_norm))])
    each_fold["tsne_f_1"] = X_norm[:, 0]
    each_fold["tsne_f_2"] = X_norm[:, 1]
    each_fold["label"] = y
    each_fold.to_excel(save_path_excel)
