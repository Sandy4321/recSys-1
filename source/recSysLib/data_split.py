import numpy as np
from sklearn.model_selection import train_test_split

def holdout(data, perc=0.99, seed=1234, clean_test=True):

    data_train, data_validation = train_test_split(data, test_size=1-perc, random_state=seed)

    print("DATA TRAIN: {}".format(type(data_train)))
    print(data_train)

    print("DATA VALID: {}".format(type(data_validation)))
    print(data_validation)

    return data_train,data_validation