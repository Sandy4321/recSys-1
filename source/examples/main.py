from recSysLib.dataset_partition import DataPartition
from recSysLib.data_split import holdout
from recSysLib.data_utils import get_urm, load_sparse_mat, store_sparse_mat
from recSysLib.metrics import precision, recall
from recSysLib import slim, content 
from recSysLib import abPredictor as abp
from recSysLib.netflix_reader import NetflixReader
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

BASELINE = '../../datasources/slim/'
USAGE = "CBF"

verbose = 1 # Not all the print are related to prints! Some are persistent.

netflix_reader = NetflixReader()

#A) INIT
# Dataset loading and partitioning
print("INIT")
netflix_urm = get_urm()
urm_partition = DataPartition(netflix_urm)
urm_partition.split_cross(train_perc_col=percentage_train_col, train_perc_row=percentage_train_row,verbose=0)

train_URMmatrix = urm_partition.get_upLeft_matrix()
test_URMmatrix = urm_partition.get_lowLeft_matrix()

icm_reduced_matrix = netflix_reader._icm_reduced_matrix

#B) WEIGHT MATRIX COMPUTATION 
print("Original: {} , train : {} , test: {} ".format(netflix_urm.shape,train_URMmatrix.shape, test_URMmatrix.shape))
model_slim = slim.MultiThreadSLIM(l1_penalty=l1,l2_penalty=l2)
model_cbf = content.Simple_CBF('Pearson')

# TO CHECK
#print("HOLDOUT")
#trainURM, validationURM = holdout(train_URMmatrix, perc = 0.70, clean_test = True)
#users_validation = validationURM.nonzero()[0]

if TO_COMPUTE_MATRIX:
    print("INIT SIM MATRIX COMPUTATION")
    model_slim.fit(train_URMmatrix)
    weight_matrix = model_slim.get_weight_matrix()
    store_sparse_mat(weight_matrix,BASELINE + 'slimW_{}_{}.npz'.format(l1,l2))

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
    print("INIT _ SIM MATRIX LOADING")
    weight_matrix_slim = load_sparse_mat(BASELINE + 'slimW_{}_{}.npz'.format(l1,l2))
    model_slim.set_urm_matrix(test_URMmatrix)
    #model_slim.set_weight_matrix(weight_matrix)
    #print(weight_matrix)
    if verbose > 0:
        print("Weight Matrix type: ",type(weight_matrix_slim))
        print("Weight Matrix shape: ",weight_matrix_slim.shape)


#C) EVALUATION
print("EVALUATION")
#print("\nTEST URM MATRIX.nonzero()",test_URMmatrix.nonzero())
#print("\nIndexes unique",np.unique(test_URMmatrix.nonzero()[0]))

n_eval = 0
metric_ = 0.0
metric_1 = 0.0
iteration = 0

abPredictor = abp.abPredictor()

# Compute residual and sampled matrices. (holdout for evaluation)
print("HOLDOUT SPLITTING")
model_slim.decompose_urm_matrix(k, verbose = 0)
# Set of learning matrix as the sampled_csc
print("PRE-EVAL")
model_slim.set_urm_matrix(model_slim.get_residual_csc())

if USAGE == "CBF":
    model_cbf.fit(icm_reduced_matrix, verbose = 0)
    model_slim.set_weight_matrix(model_cbf.get_weight_matrix())
elif USAGE == "ABP":
    model_slim.set_weight_matrix(abPredictor.get_weight_matrix())
elif USAGE == "SLIM":
    model_slim.set_weight_matrix(weight_matrix_slim)
#evaluation_URMmatrix = model_slim.get_sampled_csc()
#print(weight_matrix)


##Helper function to do a single user
def _do_user(user_to_test):
    relevant_items = evaluation_URMmatrix[user_to_test].nonzero()[1] 
    if len(relevant_items) > 0:
        recommended_items = model_slim.recommend(user_id = user_to_test,
                                                exclude_seen=True)
        prec = precision(recommended_items, relevant_items, at=k)
        rec = recall(recommended_items, relevant_items, at=k)
        return (prec, rec)
    return (np.nan, np.nan)

users_test = np.unique(test_URMmatrix.nonzero()[0])
#print(test_URMmatrix)
print("\nITERATION FOR EVALUATION")
evaluation_URMmatrix = model_slim.get_sampled_csc()
import multiprocessing
pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
res = pool.map(_do_user, (u for u in users_test))
res = np.array(res)
print(res)

print("The precision and recall are: {}".format(np.nanmean(res, axis=0)))
