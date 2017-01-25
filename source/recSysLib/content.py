import numpy as np
import scipy.sparse as sps
import scipy.stats as stats
import pickle
from .base import Recommender,check_matrix

WEIGHT_CBF = "../../datasources/matrices/cbf.pkl"
MIN_SIM = 1e-5

class Simple_CBF(Recommender):
    """
    Train a CBF algorithm. The followed pattern consists of computing the 
    weight matrix smilarity by leveraging on existing metrics such as: Pearson,
    TF-IDF ...
    For the moment only Pearson similarity is available.
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
        print("Init CBF weight matrix computation")
        if verbose > 0:
            print("ICM conversion to csc sparse matrix")
            print("ICM type: {}, shape: {}".format(type(X), X.shape))

        item_indexes = np.unique(X.nonzero()[0])

        if verbose > 0:
            print("Item Indexes {},\nlen: {}".format(item_indexes,
                                                     len(item_indexes)))

        try:
            with open(WEIGHT_CBF, 'rb') as in_file:
                self._weight_matrix = pickle.load(in_file)
            print("Load CBF weight matrix")
        except:
            self._weight_matrix = sps.lil_matrix((len(item_indexes),len(item_indexes)), dtype=np.float32)
            print(item_indexes)
            for i in item_indexes:
                tmp_sentinel = True
                if i%100==0 and tmp_sentinel:
                    tmp_sentinel = False
                    print("Iteration {} in CBF".format(i))
                for j in item_indexes:
                    if i <= j:
                        continue
                    if self.metric == 'Pearson':
                        if verbose > 0:
                            print("item i: {}\nitem j:{}".format(X[i],X[j]))
                            print("content i:{}\ncontent j:{}".format(X[i].toarray()[0],X[j].toarray()[0]))
                        c,_ = stats.pearsonr(X[i].toarray()[0], X[j].toarray()[0])
                        if verbose > 0:
                            print("item-1: {}\nitem-2:{}:sim{}".format(X[i].nonzero()[1],X[j].nonzero()[1], c))
                        
                    if c > MIN_SIM:
                        self._weight_matrix[i,j] = c
                        self._weight_matrix[j,i] = c
            if verbose > 0:
                print("Final weight matrix: {}".format(self._weight_matrix))
            
            with open(WEIGHT_CBF, 'wb') as out_file:
                pickle.dump(self._weight_matrix, out_file, pickle.HIGHEST_PROTOCOL) 
            
            print("Weight computation - CBF, END") 

    def get_weight_matrix(self):
        return self._weight_matrix
