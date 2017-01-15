import numpy as np
import scipy.sparse as sps
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
    return sps.csr_matrix((ratings, (rows, columns)), shape=shape)

def save_sparse(filename,array):
    "Store a sparse matrix by distinguish its three components: data, indices, indptr "
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse(filename,type):
    "Load a sparse matrix by distinguish its three components: data, indices, indptr"
    loader = np.load(filename)
    if type == 'csc':
        return sps.csc_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    elif type == 'csr':
        return sps.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])