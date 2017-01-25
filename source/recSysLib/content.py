import nupmy as np
import scipy.sparse as sps
import scipy.stats as stats
from .base import Recommender,check_matrix


class Simple_CBF(Recommender):
    """
    Train a CBF algorithm. The followed pattern consists of computing the 
    weight matrix smilarity by leveraging on existing metrics such as: Pearson,
    TF-IDF ...
    """
    
    # Initialization
    def __init__(self, metric ='Pearson'):
        super(Simple_CBF, self).__init__()
        self.metric = metric

    # toString()
    def __str__(self):
        return "Simple_CBF(metric ='{}')".format(self.metric)

    def fit(self, X, verbose = 0):
        self.icm = X
        X = check_matrix(X, format='csc', dtype=np.int32)

        if verbose > 0:
            print("ICM conversion to csc sparse matrix")
            print("ICM type: {}, shape: {}".format(type(X), X.shape))

        item_indexes = X.nonzero()[0]
        
        if verbose > 0:
            print("Item Indexes {},\nlen: {}".format(item_indexes,
                                                     len(item_indexes)))
    
        self.weight_matrix =
        sps.lil_matrix((len(item_indexes),len(item_indexes)), dtype=np.float32)
        for i in item_indexes[:3]:
            for j in item_indexes[:3]:
                if metric == 'Pearson':
                    c = stats.pearsonr(X[i], X[j])
                    if verbose > 0:
                        print("item-1: {}\nitem-2:{}: sim
                              {}".format(X[i], X[j], c))
                self.weight_matrix[i,j] = c

        if verbose > 0:
            print("Final weight matrix: {}".format(self.weight_matrix))

            


