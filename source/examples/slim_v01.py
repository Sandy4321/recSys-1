from recSysLib.netflix_reader import NetflixReader
from recSysLib.data_split import holdout
from recSysLib.metrics import roc_auc, precision, recall
from recSysLib import slim

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

TO_COMPUTE_SLIM = True

# Get the URM

netflixReader = NetflixReader()
netflix_urm = netflixReader.urm

if TO_COMPUTE_SLIM:

    # Split in train and validation by holdout
    sub_train , sub_validation = holdout(netflix_urm, perc=0.95, clean_test=True)

    indexes = sub_validation.nonzero()
    print("Indexes:",indexes)
    print("Indexes[0]",indexes[0])
    print("Indexes[0]unique",np.unique(indexes[0]))

    users_validation = np.unique(indexes[0])

    for i in range(-5,2,2):
        l1 = 10**i
        l2 = 0
        #l2 = numbernp.float32(random.uniform(0, 10.0))

        model = slim.MultiThreadSLIM(l1_penalty=l1,l2_penalty=l2)
        model.fit(sub_train)
        #weight_matrix = model.get_weight_matrix()

        n_eval = 0
        for user_to_test in users_validation[:3]:
            print("User: {} \n Item indices {}".format(user_to_test,sub_validation[user_to_test].indices))
            relevant_items = sub_validation[user_to_test].indices
            if len(relevant_items) > 0:
                n_eval += 1
                recommended_items = model.recommend(user_id=user_to_test, exclude_seen=True)
        #evaluate()
else:
    weight_matrix = np.load('../../datasources/slim_W01.npz','csc')

print("\n===SLIM ITEM-WEIGHT-SIMILARITY MATRIX===")
print("Shape: {}".format(weight_matrix.shape))
print("Non zero elements: {}".format(weight_matrix.getnnz()))
print("Non zero elements all cols len: {}".format(len(weight_matrix.getnnz(axis=0))))
print("Non zero elements all cols sum: {}".format(np.sum(weight_matrix.getnnz(axis=0))))
print("Non zero elements all cols avg: {}".format(np.sum(weight_matrix.getnnz(axis=0))/len(weight_matrix.getnnz(axis=0))))

