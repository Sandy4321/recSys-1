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
    def fit(self, X, verbose = 0):
        self.dataset = X

        # Conversion to a csc format [ csc = sparse matrix factorized by columns]
        X = check_matrix(X, 'csc', dtype=np.float32)
        n_items = X.shape[1] # --> X = URM (n_users x n_items)

        if verbose > 0:
            print("matrix shape {}".format(X.shape))
            print("Definition of ElasticNet")

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
            if verbose > 0:
                print("Iteration {} over {}".format(j,n_items))

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
            # Massimo version

            nnz_idx = self.model.coef_ > 0.0

            values.extend(self.model.coef_[nnz_idx])
            rows.extend(np.arange(n_items)[nnz_idx])
            cols.extend(np.ones(nnz_idx.sum()) * j)

            # finally, replace the original values of the j-th column
            X.data[startptr:endptr] = bak

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def get_weight_matrix(self):
        """
        Simple getter
        :return: get the weight matrix as a compressed sparse column matrix
        """
        return self.W_sparse

from multiprocessing import Pool
from functools import partial

class MultiThreadSLIM(Slim):
        def __init__(self,
                     l1_penalty=0.1,
                     l2_penalty=0.1,
                     positive_only=True,
                     workers=4):
            super(MultiThreadSLIM, self).__init__(l1_penalty=l1_penalty,
                                                  l2_penalty=l2_penalty,
                                                  positive_only=positive_only)
            self.workers = workers

        def __str__(self):
            return "SLIM_mt (l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
                self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
            )

        def _partial_fit(self, j, X):
            model = ElasticNet(alpha=1.0,
                               l1_ratio=self.l1_ratio,
                               positive=self.positive_only,
                               fit_intercept=False,
                               copy_X=False)
            # WARNING: make a copy of X to avoid race conditions on column j
            # TODO: We can probably come up with something better here.
            X_j = X.copy()
            # get the target column
            y = X_j[:, j].toarray()
            # set the j-th column of X to zero
            X_j.data[X_j.indptr[j]:X_j.indptr[j + 1]] = 0.0
            # fit one ElasticNet model per column
            model.fit(X_j, y)
            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nnz_idx = model.coef_ > 0.0
            values = model.coef_[nnz_idx]
            rows = np.arange(X.shape[1])[nnz_idx]
            cols = np.ones(nnz_idx.sum()) * j
            return values, rows, cols

        def fit(self, X):
            self.dataset = X
            X = check_matrix(X, 'csc', dtype=np.float32)
            n_items = X.shape[1]
            # fit item's factors in parallel
            _pfit = partial(self._partial_fit, X=X)
            pool = Pool(processes=self.workers)
            res = pool.map(_pfit, np.arange(n_items))

            # res contains a vector of (values, rows, cols) tuples
            values, rows, cols = [], [], []
            for values_, rows_, cols_ in res:
                values.extend(values_)
                rows.extend(rows_)
                cols.extend(cols_)
                # generate the sparse weight matrix

            self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)