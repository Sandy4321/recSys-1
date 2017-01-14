import numpy as np
import scipy.sparse as sps
import pandas as pd


def read_dataset(path,
                 format='csv',
                 header=None,
                 columns=None,
                 make_implicit=False,
                 implicit_th=4.0,
                 user_key='user_id',
                 item_key='item_id',
                 rating_key='rating',
                 sep=',',
                 user_to_idx=None,
                 item_to_idx=None):

    if format == 'csv':
        data = pd.read_csv(path, header=header, names=columns, sep=sep)
    elif format == 'json':
        data = pd.read_json(path)

    if make_implicit:
        data = data[data[rating_key] >= implicit_th]   # cut-threshold for the ratings
        data = data.reset_index(drop=True)   # reset the index to remove the 'holes' in the DataFrame

    if not ('item_idx' in data.columns and 'user_idx' in data.columns):
        # build user and item maps (and reverse maps)
        # this is used to map ids to indexes starting from 0 to nitems (or nusers)
        items = data[item_key].unique()
        users = data[user_key].unique()
        if item_to_idx is None:
            item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
        if user_to_idx is None:
            user_to_idx = pd.Series(data=np.arange(len(users)), index=users)
        idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
        idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)
        #  map ids to indices
        data['item_idx'] = item_to_idx[data[item_key].values].values
        data['user_idx'] = user_to_idx[data[user_key].values].values
        return data, idx_to_user, idx_to_item
    else:
        return data


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
