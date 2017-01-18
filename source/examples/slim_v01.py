from recSysLib.dataset_partition import DataPartition
from recSysLib.data_split import holdout
from recSysLib.data_utils import get_urm
from recSysLib.metrics import precision, recall
from recSysLib import slim

import numpy as np

TO_COMPUTE_SLIM = True


# A) Get the URM
netflix_urm = get_urm()

# B) Split in Train and Test
percentage_train_col = 0.8
percentage_train_row = 0.4

netflix_urm = get_urm()
urm_partition = DataPartition(netflix_urm)
urm_partition.split_train_test(train_perc_col=percentage_train_col, train_perc_row=percentage_train_row,verbose=0)

train_URMmatrix = urm_partition.get_train_matrix()
test_URMmatrix = urm_partition.get_test_matrix()

# Further reduction for l1 and l2 estimation
train_partition = DataPartition(train_URMmatrix)
train_partition.split_train_test(train_perc_row=percentage_train_row,train_perc_col=percentage_train_col, verbose=0)
add_train_urm = train_partition.get_train_matrix()

print("Original: {} , train : {} , validate-l1-l2: {} ".format(netflix_urm.shape,train_URMmatrix.shape, add_train_urm.shape))
if TO_COMPUTE_SLIM:
    # Split in train and validation by holdout
    sub_train , sub_validation = holdout(add_train_urm, perc=0.95, clean_test=True)

    print("SHAPES Sub Train: {}, sub Validation: {}".format(sub_train.shape, sub_validation.shape))
    print("LENGTHS Sub Train: {}, sub Validation: {}".format(len(sub_train.nonzero[0]), len(sub_validation.nonzero[0])))
    indexes = sub_validation.nonzero()

    users_validation = np.unique(indexes[0])

    iteration = 0
    for i in range(-1,4,2):
        iteration += 1
        l2 = 10**i
        l1 = 10
        #l2 = numbernp.float32(random.uniform(0, 10.0))

        # Compute the model with the established parameters
        model = slim.MultiThreadSLIM(l1_penalty=l1,l2_penalty=l2)
        model.fit(sub_train)
        weight_matrix = model.get_weight_matrix()
        print("SLIM similarity nnz: ",len(weight_matrix.nonzero()[0]))
        print("weight-matrix:",weight_matrix)
        # Evaluation
        n_eval = 0
        metric_ = 0.0
        metric_1 = 0.0
        for user_to_test in users_validation:
            #print("User: {} \n Item indices {}".format(user_to_test,sub_validation[user_to_test].indices))
            relevant_items = sub_validation[user_to_test].indices
            if len(relevant_items) > 0:
                n_eval += 1
                recommended_items = model.recommend(user_id=user_to_test, exclude_seen=True)
                metric_ += precision(recommended_items, relevant_items)
                metric_1 += recall(recommended_items, relevant_items)
        metric_ /= n_eval
        metric_1 /= n_eval

        print("Iteration {} , l1 coeff: {}, precision: {}, recall : {}".format(iteration,l1,metric_,metric_1))
else:
    weight_matrix = np.load('../../datasources/slim_W01.npz','csc')
