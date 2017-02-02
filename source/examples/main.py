from recSysLib.dataset_partition import DataPartition
from recSysLib.data_split import holdout
from recSysLib.data_utils import get_urm, load_sparse_mat, store_sparse_mat
from recSysLib.metrics import precision, recall

# Models used to compute the Weight Matrix
from recSysLib import slim, content 
from recSysLib import abPredictor as abp

# Dataset reader and Evaluator class
from recSysLib.netflix_reader import NetflixReader
from recSysLib.Evaluator import Evaluator
import numpy as np
import scipy

# Run Parameters
percentage_train_col = 0.75
percentage_train_row = 0.75

percentage_sub_train_col = 0.7
percentage_sub_train_row = 0.7

k = 5 # Number of k-items to make the evaluation on
n_simil_items = 50 #number of similarities to keep for each item

l1 = 0.1
l2 = 100000

USAGE = "CBF"
TESTING = "NEW_ITEM"
CBF_METRIC = "Pearson"
IDF = False

verbose = 1 # Not all the print depend from verbose! Some are persistent.

netflix_reader = NetflixReader()

#A) INIT STAGE
# Dataset loading and partitioning: TRAIN and TEST
print("INIT")
netflix_urm = get_urm()
urm_partition = DataPartition(netflix_urm)
urm_partition.split_cross(train_perc_col=percentage_train_col, train_perc_row=percentage_train_row,verbose=0)

if TESTING == 'NEW_USER':
    train_URMmatrix = urm_partition.get_upLeft_matrix()
    test_URMmatrix = urm_partition.get_lowLeft_matrix()
elif TESTING == 'NEW_ITEM':
    train_URMmatrix = urm_partition.get_upLeft_matrix()
    test_URMmatrix = urm_partition.get_upRight_matrix()

icm_reduced_matrix = netflix_reader._icm_reduced_matrix

print("Original: {} , train : {} , test: {} ".format(netflix_urm.shape, train_URMmatrix.shape, test_URMmatrix.shape))

#B) MODEL SELECTION STAGE
if USAGE == "SLIM":
    model = slim.MultiThreadSLIM(train_URMmatrix, l1_penalty=l1,l2_penalty=l2)
elif USAGE == "CBF":
    idf_array = netflix_reader.get_idf_array()
    if IDF:
        model = content.Simple_CBF(X = icm_reduced_matrix, idf_array= idf_array, metric = CBF_METRIC, IDF = IDF)
    else:
        model = content.Simple_CBF(X = icm_reduced_matrix, metric = CBF_METRIC, IDF = IDF) 
elif USAGE == "ABP":
    model = abp.abPredictor()


#C) EVALUATION with HOLDOUT STAGE
print("EVALUATION")
weight_matrix = model.get_weight_matrix()
evaluator = Evaluator(test_URMmatrix, weight_matrix)

# Compute residual and sampled matrices. (holdout for evaluation)
if TESTING == 'NEW_USER':
    evaluator.holdout(verbose = 1)
    # Set of learning matrix as the sampled_csc
    evaluator.set_urm_matrix(evaluator.get_residual_csc())
    users_test = np.unique(test_URMmatrix.nonzero()[0])
    evaluation_URMmatrix = evaluator.get_sampled_csc()

    #cut weight matrix to the proper size
    if weight_matrix.shape[0] > test_URMmatrix.shape[1]:
        print("Cutting weight matrix to test items")
        weight_matrix = weight_matrix[:test_URMmatrix.shape[1],
                                      :test_URMmatrix.shape[1]]
    evaluator.set_weight_matrix(weight_matrix)

#use up left to predict up right
elif TESTING == 'NEW_ITEM':
    #fill the train urm with zeros on th?!?jedi=0, e right?!? (*_*arg1*_*, shape=None, dtype=None, copy=False) ?!?jedi??
    zeros_right = scipy.sparse.csc_matrix((train_URMmatrix.shape[0],
                                          test_URMmatrix.shape[1]))
    num_old_items = train_URMmatrix.shape[1]
    train_URMmatrix = scipy.sparse.hstack([train_URMmatrix, zeros_right])

    evaluator.set_urm_matrix(train_URMmatrix)
    users_test = np.unique(train_URMmatrix.nonzero()[0])
    evaluation_URMmatrix = test_URMmatrix
    #set the "starting" urm. We really need to modify the evaluator
    evaluator.residual_csc = train_URMmatrix

# iterate over each column and keep only the top-k similar items
weight_matrix=weight_matrix.toarray()
idx_sorted = weight_matrix.argsort(axis=0) # sort by column
values, rows, cols = [], [], []
nitems = weight_matrix.shape[0]
for i in range(nitems):
    top_k_idx = idx_sorted[-n_simil_items:, i]
    values.extend(weight_matrix[top_k_idx, i])
    rows.extend(np.arange(nitems)[top_k_idx])
    cols.extend(np.ones(n_simil_items) * i)
weight_matrix = scipy.sparse.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)
evaluator.set_weight_matrix(weight_matrix)

filter_old = TESTING == 'NEW_ITEM'
if filter_old:
    evaluator.set_num_old_items(num_old_items)

##Helper function to do a single user
def _do_user(user_to_test):
    relevant_items = evaluation_URMmatrix[user_to_test].nonzero()[1]
    #if we are doing nre items their ID has been moved to the left
    relevant_items += num_old_items

    if len(relevant_items) > 0:
        recommended_items = evaluator.recommend(user_id = user_to_test,
                                                exclude_seen=True,
                                                exclude_old=filter_old)
        prec = precision(recommended_items, relevant_items, at=k)
        rec = recall(recommended_items, relevant_items, at=k)
        return (prec, rec)
    return (np.nan, np.nan)


import multiprocessing
pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
res = pool.map(_do_user, (u for u in users_test))
res = np.array(res)

print("The precision and recall are: {}".format(np.nanmean(res, axis=0)))

