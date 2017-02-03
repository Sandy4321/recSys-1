from recSysLib.dataset_partition import DataPartition
from recSysLib.data_split import holdout
from recSysLib.data_utils import get_urm, load_sparse_mat, store_sparse_mat
from recSysLib.metrics_max import precision, recall, roc_auc, rr, ndcg, map

# Models used to compute the Weight Matrix
from recSysLib import slim, content_sim 
from recSysLib import abPredictor as abp

# Dataset reader and Evaluator class
from recSysLib.netflix_reader import NetflixReader
from recSysLib.Evaluator import Evaluator
import numpy as np
import scipy
import sys

# Run Parameters
percentage_train_col = 0.75
percentage_train_row = 0.75

percentage_sub_train_col = 0.7
percentage_sub_train_row = 0.7

k1 = 1 # Number of k-items to make the evaluation on
k2 = 2
k3 = 5
k4 = 10
k5 = 25
k6 = 50

l1 = 0.1
l2 = 100000

USAGE = sys.argv[1]
TESTING = sys.argv[2]
CBF_METRIC = sys.argv[3]
IDF = sys.argv[4] == '1'
n_simil_items = int(sys.argv[5]) #number of similarities to keep for each item
SHRINK = int(sys.argv[6])

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
icm_idf_matrix = scipy.sparse.lil_matrix(icm_reduced_matrix.shape)

print("Original: {} , train : {} , test: {} ".format(netflix_urm.shape, train_URMmatrix.shape, test_URMmatrix.shape))

#B) MODEL SELECTION STAGE
if USAGE == "SLIM":
    model = slim.MultiThreadSLIM(train_URMmatrix, l1_penalty=l1,l2_penalty=l2)
elif USAGE == "CBF":
    idf_array = netflix_reader.get_idf_array()
    if IDF:
        for row in range(icm_reduced_matrix.shape[0]):
            icm_idf_matrix[row] =icm_reduced_matrix[row].multiply(idf_array)[0]
        model = content_sim.Simple_CBF(X = icm_idf_matrix.T, metric =
                                       CBF_METRIC, IDF = IDF, shrink=SHRINK)
    else:
        model = content_sim.Simple_CBF(X = icm_reduced_matrix.T, metric =
                                       CBF_METRIC, IDF = IDF, shrink = SHRINK)
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
    #fill the train urm with zeros on the right
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
if not isinstance(weight_matrix, np.ndarray):
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
    #if we are doing new items their ID has been moved to the left
    if filter_old:
        relevant_items += num_old_items

    if len(relevant_items) > 0:
        recommended_items = evaluator.recommend(user_id = user_to_test,
                                                exclude_seen=True,
                                                exclude_old=filter_old)
        auc = roc_auc(recommended_items, relevant_items)
        
        prec1 = precision(recommended_items, relevant_items, at=k1)
        prec2 = precision(recommended_items, relevant_items, at=k2)
        prec3 = precision(recommended_items, relevant_items, at=k3)
        prec4 = precision(recommended_items, relevant_items, at=k4)
        prec5 = precision(recommended_items, relevant_items, at=k5)
        prec6 = precision(recommended_items, relevant_items, at=k6)

        rec1 = recall(recommended_items, relevant_items, at=k1)
        rec2 = recall(recommended_items, relevant_items, at=k2)
        rec3 = recall(recommended_items, relevant_items, at=k3)
        rec4 = recall(recommended_items, relevant_items, at=k4)
        rec5 = recall(recommended_items, relevant_items, at=k5)
        rec6 = recall(recommended_items, relevant_items, at=k6)
       
        rr1 = rr(recommended_items, relevant_items, at=k1)
        rr2 = rr(recommended_items, relevant_items, at=k2)
        rr3 = rr(recommended_items, relevant_items, at=k3)
        rr4 = rr(recommended_items, relevant_items, at=k4)
        rr5 = rr(recommended_items, relevant_items, at=k5)
        rr6 = rr(recommended_items, relevant_items, at=k6)

        map1 = map(recommended_items, relevant_items, at=k1)
        map2 = map(recommended_items, relevant_items, at=k2)
        map3 = map(recommended_items, relevant_items, at=k3)
        map4 = map(recommended_items, relevant_items, at=k4)
        map5 = map(recommended_items, relevant_items, at=k5)
        map6 = map(recommended_items, relevant_items, at=k6)

        ndcg1 = ndcg(recommended_items, relevant_items, at=k1)
        ndcg2 = ndcg(recommended_items, relevant_items, at=k2)
        ndcg3 = ndcg(recommended_items, relevant_items, at=k3)
        ndcg4 = ndcg(recommended_items, relevant_items, at=k4)
        ndcg5 = ndcg(recommended_items, relevant_items, at=k5)
        ndcg6 = ndcg(recommended_items, relevant_items, at=k6)

        return (auc, prec1, prec2, prec3, prec4, prec5, prec6, rec1, rec2,
                rec3, rec4, rec5, rec6, rr1, rr2, rr3, rr4, rr5, rr6, map1,
                map2, map3, map4, map5, map6, ndcg1, ndcg2, ndcg3, ndcg4,
                ndcg5, ndcg6)
    return [np.nan]*31


import multiprocessing
pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
res = pool.map(_do_user, (u for u in users_test))
res = np.array(res)

(auc, prec1, prec2, prec3, prec4, prec5, prec6, rec1, rec2, rec3, rec4, rec5,
rec6, rr1, rr2, rr3, rr4, rr5, rr6, map1, map2, map3, map4, map5, map6, ndcg1,
ndcg2, ndcg3, ndcg4,ndcg5, ndcg6) = np.nanmean(res, axis=0)

print()                                                                 
print("Results of %s on %s with %d similarities." % (USAGE,TESTING,n_simil_items))
if USAGE == 'CBF':
    print("Metric = %s, IDF = %s, SHRINK = %d" % (CBF_METRIC, IDF, SHRINK))

print("Precision@%d\tPrecision@%d\tPrecision@%d\tPrecision@%d\tPrecision@%d\tPrecision@%d"
      % (k1,k2,k3,k4,k5,k6))
print("%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f" % (prec1*100, prec2*100,
                                                        prec3*100, prec4*100,
                                                        prec5*100, prec6*100))

print()
print("Recall@%d\tRecall@%d\tRecall@%d\tRecall@%d\tRecall@%d\tRecall@%d"
      % (k1,k2,k3,k4,k5,k6))
print("%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f" % (rec1*100, rec2*100, rec3*100,
                                                        rec4*100, rec5*100, rec6*100))

print()
print("RR@%d\t\tRR@%d\t\tRR@%d\t\tRR@%d\t\tRR@%d\t\tRR@%d"
      % (k1,k2,k3,k4,k5,k6))
print("%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f" % (rr1*100, rr2*100, rr3*100,
                                                        rr4*100, rr5*100, rr6*100))

print()
print("Map@%d\t\tMap@%d\t\tMap@%d\t\tMap@%d\t\tMap@%d\t\tMap@%d"
      % (k1,k2,k3,k4,k5,k6))
print("%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f" % (map1*100, map2*100, map3*100,
                                                        map4*100, map5*100, map6*100))

print()
print("NDCG@%d\t\tNDCG@%d\t\tNDCG@%d\t\tNDCG@%d\t\tNDCG@%d\t\tNDCG@%d"
      % (k1,k2,k3,k4,k5,k6))
print("%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (ndcg1, ndcg2, ndcg3,
                                                        ndcg4, ndcg5, ndcg6))


print()
print("AUC:%.4f" % (auc))

