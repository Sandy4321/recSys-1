import numpy as np
import scipy.sparse as sps
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from .base import Recommender,check_matrix

MIN_SIM = 1e-5

class Simple_CBF(Recommender):
    """
    Train a CBF algorithm. The followed pattern consists of computing the 
    weight matrix smilarity by leveraging on existing metrics such as: Pearson,
    TF-IDF ...
    For the moment only Pearson similarity is available.
    """
    
    # Initialization
    def __init__(self, X, metric ='Pearson', idf_array = None, IDF = True):
        super(Simple_CBF, self).__init__()
        self.metric = metric
        self.icm = X
        # Weight matrix computation
        if IDF:
            self._idf_array = idf_array
        self.idf_mode = IDF

        self.WEIGHT_CBF = "../../datasources/matrices/cbf_"
        self.WEIGHT_CBF += metric
        if IDF:
            self.WEIGHT_CBF += "_IDF_"
        else:
            self.WEIGHT_CBF += "_bin_"

        self.WEIGHT_CBF += ".pkl"

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
        try:
            with open(self.WEIGHT_CBF, 'rb') as in_file:
                self._weight_matrix = pickle.load(in_file)
            print("Load CBF weight matrix")
        except:
            print("Compute CBF")
            self._weight_matrix = sps.lil_matrix((len(item_indexes),len(item_indexes)), dtype=np.float32)
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
                        print("TYPES: X[i] {}, X[i].toarray() {}, X[i].toarray()[0] {}".format( type(X[i]), type(X[i].toarray()),type(X[i].toarray()[0])))
                        print("TYPE IDF",type(self._idf_array) ,
                              self._idf_array.shape)
                        print("SHAPES", (X[i].toarray()[0]).shape)
                        print("RESHAPES", type(X[i].toarray()[0].reshape(4699,1)), X[i].toarray()[0].reshape(-1).shape)
                        print("TO LIST",
                              type(np.asarray(X[i].toarray()[0].tolist())),
                              np.asarray(X[i].toarray()[0].tolist()).shape)
#                        print("To ARRAY TEST",X[i].toarray()[0])
#                        print("MUL TEST",(X[i].toarray()[0]).multiply(self._idf_array))
                        print("MULTIPLY",X[i].multiply(self._idf_array).shape,
                              type(np.array(X[i].multiply(self._idf_array))))

                        if self.idf_mode == True: 
                            c,_ = stats.pearsonr(np.multiply(X[i], self._idf_array), np.multiply(X[j], self._idf_array))
                            #print(type(X[i].toarray()[0]))
                            #print("Shapes ELSE:",X[i].toarray()[0].shape)
                            #print("Shapes IF:",np.dot(X[i].toarray()[0], self._idf_array).shape)
                        else:
                            c,_ = stats.pearsonr(X[i].toarray()[0], X[j].toarray()[0])

                        if verbose > 0:
                            print("item-1: {}\nitem-2:{}:sim{}".format(X[i].nonzero()[1],X[j].nonzero()[1], c))
                    if self.metric == 'Cosine':
                        if verbose > 0:
                            print("item i: {}\nitem j:{}".format(X[i],X[j]))
                            print("shapes: {}, {}".format(X[i].shape, X[j].shape))
                        
                        if self.idf_mode == True:
                            c = cosine_similarity(X[i].multiply(self._idf_array), X[j].multiply(self._idf_array))[0][0] 
                        else:
                            c = cosine_similarity(X[i], X[j])[0][0]
                    if c > MIN_SIM:
                        self._weight_matrix[i,j] = c
                        self._weight_matrix[j,i] = c
            if verbose > 0:
                print("Final weight matrix: {}".format(self._weight_matrix))
            
            with open(self.WEIGHT_CBF, 'wb') as out_file:
                pickle.dump(self._weight_matrix, out_file, pickle.HIGHEST_PROTOCOL) 
            
            print("Weight computation - CBF, END") 

    def get_weight_matrix(self):
        return self._weight_matrix
