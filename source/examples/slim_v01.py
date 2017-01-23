from recSysLib.dataset_partition import DataPartition
from recSysLib.data_split import holdout
from recSysLib.data_utils import get_urm, load_sparse_mat, store_sparse_mat
from recSysLib.metrics import precision, recall
from recSysLib import slim

import numpy as np


# Run Parameters
TO_COMPUTE_MATRIX = False
percentage_train_col = 0.75
percentage_train_row = 0.75

percentage_sub_train_col = 0.7
percentage_sub_train_row = 0.7

k = 5 # Number of k-items to make the evaluation on

l1 = 0.1
l2 = 100000

#A) INIT
# Dataset loading and partitioning
print("INIT")
netflix_urm = get_urm()
urm_partition = DataPartition(netflix_urm)
urm_partition.split_cross(train_perc_col=percentage_train_col, train_perc_row=percentage_train_row,verbose=0)

train_URMmatrix = urm_partition.get_upLeft_matrix()
test_URMmatrix = urm_partition.get_lowLeft_matrix()

#B) WEIGHT MATRIX COMPUTATION 
print("Original: {} , train : {} , test: {} ".format(netflix_urm.shape,train_URMmatrix.shape, test_URMmatrix.shape))
model = slim.MultiThreadSLIM(l1_penalty=l1,l2_penalty=l2)

# TO CHECK
#print("HOLDOUT")
#trainURM, validationURM = holdout(train_URMmatrix, perc = 0.70, clean_test = True)
#users_validation = validationURM.nonzero()[0]

if TO_COMPUTE_MATRIX:
    print("INIT SIM MATRIX COMPUTATION")
    model.fit(train_URMmatrix)
    weight_matrix = model.get_weight_matrix()
    store_sparse_mat(weight_matrix,'../../datasorces/slimW_{}_{}.npz'.format(l1,l2))

    #print("SLIM similarity nnz: ",len(weight_matrix.nonzero()[0]))
    #print("weight-matrix:",weight_matrix)
    # Evaluation
    #n_eval = 0
    #metric_ = 0.0
    #metric_1 = 0.0
    #for user_to_test in users_validation:
         #print("User: {} \n Item indices {}".format(user_to_test,sub_validation[user_to_test].indices))
    #    relevant_items = sub_validation[user_to_test].indices
    #    if len(relevant_items) > 0:
    #        n_eval += 1
    #        recommended_items = model.recommend(user_id=user_to_test, exclude_seen=True)
    #        metric_ += precision(recommended_items, relevant_items, at=k)
    #        metric_1 += recall(recommended_items, relevant_items, at=k)
    #metric_ /= n_eval
    #metric_1 /= n_eval

    #print("Iteration {} , l1 - l2 coeff: {}@{}, precision: {}, recall : {}".format(iteration,l1,l2,metric_,metric_1))

else:
    print("INIT SIM MATRIX LOADING")
    weight_matrix = load_sparse_mat('../../datasources/slimW_{}_{}.npz'.format(l1,l2))
    model.set_urm_matrix(test_URMmatrix)
    model.set_weight_matrix(weight_matrix)
    #print(weight_matrix)
    print("Weight Matrix type: ",type(weight_matrix))
    print("Weight Matrix shape: ",weight_matrix.shape)


#C) EVALUATION
print("EVALUATION")
#print("\nTEST URM MATRIX.nonzero()",test_URMmatrix.nonzero())
#print("\nIndexes unique",np.unique(test_URMmatrix.nonzero()[0]))

n_eval = 0
metric_ = 0.0
metric_1 = 0.0
iteration = 0

# Compute residual and sampled matrices. (holdout for evaluation)
print("DECOMPOSITION")
model.decompose_urm_matrix(k, verbose = 1)
# Set of learning matrix as the sampled_csc
print("PRE-EVAL")
model.set_urm_matrix(model.get_sampled_csc())
#print(weight_matrix)

users_test = np.unique(test_URMmatrix.nonzero()[0])
#print(test_URMmatrix)
print("\nITERATION FOR EVALUATION")
evaluation_URMmatrix = model.get_residual_csc()
for user_to_test in users_test:
    iteration += 1
    if iteration%100 == 0:
        print("Iteration: {} over {}".format(iteration, len(users_test)))
    relevant_items = evaluation_URMmatrix[user_to_test].nonzero()[1]
    #print(test_URMmatrix[user_to_test])
    if len(relevant_items) > 2*k:
        n_eval += 1
        #print("SHAPES: train_URM {}, test_URM {}, weight {}, test_URM[u] {}".format(train_URMmatrix.shape, test_URMmatrix.shape, weight_matrix.shape, test_URMmatrix[user_to_test].shape))
        recommended_items = model.recommend(user_id = user_to_test, exclude_seen=False)
        #print("relevant items: {}\nrecommended items:{}\n".format(relevant_items, recommended_items))
        metric_ += precision(recommended_items, relevant_items, at=k)
        metric_1 += recall(recommended_items, relevant_items, at=k)
        #print(metric_,metric_1)
metric_ /= n_eval
metric_1 /= n_eval
print("The precision is {} and the recall: {}".format(metric_,metric_1))
