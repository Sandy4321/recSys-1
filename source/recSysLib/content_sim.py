import numpy as np
import scipy.sparse as sps
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from .base import Recommender,check_matrix
import recpy.recommenders.similarity as sim 

MIN_SIM = 1e-5

class Simple_CBF(Recommender):
    """
    Train a CBF algorithm. The followed pattern consists of computing the
    weight matrix smilarity by leveraging on existing metrics such as: Pearson,
    TF-IDF ...
    For the moment only Pearson similarity is available.
    """

    # Initialization
    def __init__(self, X, metric ='Pearson', idf_array = None, IDF = True,
                 shrink = 10):
        super(Simple_CBF, self).__init__()
        self.metric = metric
        self.icm = X
        self.shrink = shrink
        # Weight matrix computation
        if IDF:
            self._idf_array = idf_array
        self.idf_mode = IDF

        self._compute_weight_matrix(verbose = 0)

    # toString()
    def __str__(self):
        return "Simple_CBF(metric ='{}')".format(self.metric)

    def _compute_weight_matrix(self, verbose = 0):
        X = check_matrix(self.icm, format='csc', dtype=np.int32)
        print("Init CBF weight matrix computation")
        print(self.metric + " IDF: " + str(self.idf_mode))
        if verbose > 0:
            print("ICM conversion to csc sparse matrix")
            print("ICM type: {}, shape: {}".format(type(X), X.shape))

        n_items = X.shape[0]
        item_indexes = [i for i in range(n_items-1,-1,-1)]
        if verbose >1:
            print("Item Indexes {},\nlen: {}".format(item_indexes,
                                                     len(item_indexes)))

        self.WEIGHT_CBF = "../../datasources/matrices/cbf_"
        self.WEIGHT_CBF += self.metric
        if self.idf_mode:
            self.WEIGHT_CBF += "_IDF_"
        else:
            self.WEIGHT_CBF += "_bin_"

        self.WEIGHT_CBF += str(self.shrink) + '.pkl'

        try:
            with open(self.WEIGHT_CBF, 'rb') as in_file:
                print("Init loading CBF weight matrix")
                self._weight_matrix = pickle.load(in_file)
            print("Load CBF weight matrix")
        except:
            print("Init computation CBF")
            self._weight_matrix = sps.lil_matrix((len(item_indexes),len(item_indexes)), dtype=np.float32)


            if self.metric == 'Pearson':
                if verbose > 0:
                    print("item i: {}\nitem j:{}".format(X[i],X[j]))
                    print("content i:{}\ncontent j:{}".format(X[i].toarray()[0],X[j].toarray()[0]))
                    print("item-1: {}\nitem-2:{}:sim{}".format(X[i].nonzero()[1],X[j].nonzero()[1], c))

                self._weight_matrix = sim.Pearson(shrinkage = self.shrink).compute(X)

            if self.metric == 'Cosine':
                if verbose > 0:
                    print("item i: {}\nitem j:{}".format(X[i],X[j]))
                    print("shapes: {}, {}".format(X[i].shape, X[j].shape))

                self._weight_matrix = sim.Cosine(shrinkage = self.shrink).compute(x)

            if verbose > 0:
                print("Final weight matrix: {}".format(self._weight_matrix))

            with open(self.WEIGHT_CBF, 'wb') as out_file:
                pickle.dump(self._weight_matrix, out_file, pickle.HIGHEST_PROTOCOL)

            print("Weight computation - CBF, END")
            return self._weight_matrix

    def get_weight_matrix(self):
        return self._weight_matrix
