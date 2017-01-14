import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from .base import Recommender,check_matrix

# extends the abstract class Recommender, it is contained in the file base.py
class Slim(Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    # Initialization of the slim class
    def __init__(self, l1_penalty=0.1, l2_penalty=0.1, positive_only=True):
        # According to the paper ElasticNet notatoin: l1_penalty = a ; l2_penalty = b.
        super(Slim, self).__init__()
        self.l1_penalty = l1_penalty # penalty associated with the norm-1
        self.l2_penalty = l2_penalty # penalty associated with the norm-2
        self.positive_only = positive_only # constraint about restriction to only positive weights
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)

    # Equivalent to the toString method.
    def __str__(self):
        return "SLIM (l1_penalty={},l2_penalty={},positive_only={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only
        )

    # Fit method, it computes the weight matrix by solving the "optimization problem"
    def fit(self, X):
        self.dataset = X

        # Conversion to a csc format [ csc = sparse matrix factorized by columns]
        X = check_matrix(X, 'csc', dtype=np.float32)
        n_items = X.shape[1] # --> X = URM (n_users x n_items)

        # initialize the ElasticNet model to solve the optimization problem
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False)

        # we'll store the W matrix into a sparse csc_matrix, thanks to the independence condition between columns
        # let's initialize the vectors used by the sparse.csc_matrix constructor
        values, rows, cols = [], [], []

        # fit each item's factors sequentially (not in parallel)
        for j in range(n_items):
            # get the target column corresponded to the item j
            y = X[:, j].toarray()

            # set the j-th column of X to zero
            startptr = X.indptr[j]  # index pointer array of the matrix
            endptr = X.indptr[j + 1]
            # sparse values written in the column j
            bak = X.data[startptr: endptr].copy()
            X.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            self.model.fit(X, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nnz_idx = self.model.coef_ > 0.0
            values.extend(self.model.coef_[nnz_idx])
            rows.extend(np.arange(n_items)[nnz_idx])
            cols.extend(np.ones(nnz_idx.sum()) * j)

            # finally, replace the original values of the j-th column
            X.data[startptr:endptr] = bak

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
