import numpy as np

def holdout(data, user_key='user_id', item_key='item_id', perc=0.8, seed=1234, clean_test=True):
    # set the random seed
    rng = np.random.RandomState(seed)
    #  shuffle data
    nratings = data.shape[0]
    shuffle_idx = rng.permutation(nratings)
    train_size = int(nratings * perc)
    # split data according to the shuffled index and the holdout size
    train_split = data[shuffle_idx[:train_size]]
    test_split = data[shuffle_idx[train_size:]]

    print("TRAIN SPLIT len {}\n {}",len(train_split),train_split)
    print("\nTEST SPLIT len {}:\n",len(test_split),test_split)

    # remove new user and items from the test split
    if clean_test:
        train_users = train_split[user_key].unique()
        train_items = train_split[item_key].unique()
        test_split = test_split[(test_split[user_key].isin(train_users)) & (test_split[item_key].isin(train_items))]

    return train_split, test_split
