import numpy as np
from scipy import sparse
import joblib as jbl

def df_to_csr(df, nrows, ncols, is_implicit=False, user_key='user_idx', item_key='item_idx', rating_key='rating'):
    """
    Convert a pandas DataFrame to a scipy.sparse.csr_matrix
    """

    rows = df[user_key].values
    columns = df[item_key].values
    ratings = df[rating_key].values if not is_implicit else np.ones(df.shape[0])
    shape = (nrows, ncols)
    # using the 4th constructor of csr_matrix
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    return sparse.csr_matrix((ratings, (rows, columns)), shape=shape)

def store_sparse_mat(m, name):
    jbl.dump(m,name)

def load_sparse_mat(name):
    m = jbl.load(name)
    return m