import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from base import Recommender,check_matrix
import random
import pickle

SAMPLED_CSC = "../../datasources/matrices/sampled.pkl"
RESIDUAL_CSC = "../../datasources/matrices/residual.pkl"

# extends the abstract class Recommender, it is contained in the file base.py

class Evaluator():
    def __init__(self, urm, w):
        self.dataset = urm
        self.W_sparse = w

    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict_rates(self, user_id, exclude_seen=True, verbose = 0):
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        if verbose > 0:
            print(scores)

    def set_weight_matrix(self, W):
        self.W_sparse = W

    def set_urm_matrix(self, X):
        self.dataset = X

    def get_sampled_csc(self):
        return self.sampled_csc

    def get_residual_csc(self):
        return self.residual_csc

    def decompose_urm_matrix(self, k, verbose = 0):
        #try to load the matrices, otherwise compute and store them
        try:
            with open(SAMPLED_CSC, 'rb') as infile:
                self.sampled_csc = pickle.load(infile)
            with open(RESIDUAL_CSC, 'rb') as infile:
                self.residual_csc = pickle.load(infile)
            print("Loaded sampled and residual matrices")
        except:
            self.original_dataset = self.dataset
            iteration = 0
            list_users = np.unique(self.dataset.nonzero()[0])
            csc_residual_dataset = sps.lil_matrix(self.dataset.shape)
            csc_sampled_dataset = sps.lil_matrix(self.dataset.shape)
            len_test = len(list_users)
            for u in list_users:
                iteration += 1
                user_profile = self.dataset[u,:]
                if verbose > 0 and iteration%500 == 0:
                    print("\nIteration {} over {}".format(iteration, len_test))
                    #print("Profile:",user_profile)
                rated_items = list(user_profile.nonzero()[1])
                if len(rated_items) > k:
                    sampled_items = random.sample(rated_items,k)
                    residual_items = list(set(rated_items) - set(sampled_items))
                    if verbose > 1:
                        print("All items",rated_items)
                        print("Sampled items type: {}, {}".format(type(sampled_items),sampled_items))
                        print("Residual items ",residual_items)
                    for i in residual_items:
                        csc_residual_dataset[u,i] = user_profile[0,i]

                    for i in sampled_items:
                        csc_sampled_dataset[u,i] = user_profile[0,i]
                        #print(csc_sampled_dataset[u,i])
                    #print("Sampled type:{}, {}".format(type(sampled_items), sampled_items))
            if verbose > 0:
                print("CSC sampled:\n",csc_sampled_dataset)
                print("CSC residual:\n",csc_residual_dataset)
            self.sampled_csc = csc_sampled_dataset
            self.residual_csc = csc_residual_dataset

            #and store them
            with open(SAMPLED_CSC, 'wb') as outfile:
                pickle.dump(self.sampled_csc, outfile, pickle.HIGHEST_PROTOCOL)
            with open(RESIDUAL_CSC, 'wb') as outfile:
                pickle.dump(self.residual_csc, outfile, pickle.HIGHEST_PROTOCOL)
            print("Saved sampled and residual matrices")
            print("Saved sampled and residual matrices")
