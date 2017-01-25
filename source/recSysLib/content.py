import numpy as np
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

        item_indexes = X.nonzero()[1]
        
        if verbose > 0:
            print("Item Indexes {},\nlen: {}".format(item_indexes,
                                                     len(item_indexes)))
    
        self.weight_matrix = sps.lil_matrix((len(item_indexes),len(item_indexes)), dtype=np.float32)
        for i in item_indexes:
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
                self.weight_matrix[i,j] = c
                self.weight_matrix[j,i] = c
        if verbose > 0:
            print("Final weight matrix: {}".format(self.weight_matrix))

    def get_weight_matrix(self):
        return self.weight_matrix

    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        ranking = scores.argsort()[::-1]
                                    
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        
        return ranking[:n]

    def predict_rates(self, user_id, exclude_seen=True, verbose=0):
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        if verbose > 0:
            print(scores)
