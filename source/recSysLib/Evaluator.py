import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from base import Recommender,check_matrix
import random
import pickle

SAMPLED_CSC = "../../datasources/matrices/sampled"
RESIDUAL_CSC = "../../datasources/matrices/residual"

class Evaluator:
    def __init__(self, urm, w):
        self.dataset = urm
        self.W_sparse = w

    def recommend(self, user_id, n=None, exclude_seen=True, exclude_old=False, exclude_popular=False):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id, mode = "RESIDUAL")
        scores = user_profile.dot(self.W_sparse)
        try:
            scores = scores.toarray()
        except:
            #Who cares, W was not sparse
            pass
        scores = scores.ravel()
        if exclude_old:
            scores = self._filter_old(scores)
        if exclude_popular:
            scores = self._filter_pop(scores)

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

    def _get_user_ratings(self, u, mode = "RESIDUAL"):
        if mode == "RESIDUAL":
            return self.residual_csc[u]
        elif mode == "SAMPLED":
            return self.sampled_csc[u]
        elif mode == "ORIGINAL":
            return self.original_dataset[u]

    def _filter_seen(self, user_id, ranking):
        user_profile = self._get_user_ratings(user_id, "RESIDUAL")
        #seen = user_profile.indices
        seen = list(user_profile.nonzero()[1])
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

    def _filter_old(self, scores):
        scores[:self._num_old_items] = 0 
        return scores

    def set_num_old_items(self, num):
        self._num_old_items = num

    def _filter_pop(self, scores):
        scores[self._popular_items] = 0
        return scores
    
    def set_idx_top_pop(self, idx)
        self._popular_items = idx
    
    def holdout_user(self, k, verbose = 0):
        #try to load the matrices, otherwise compute and store them
        try:
            with open(SAMPLED_CSC + '_user.pkl', 'rb') as infile:
                self.sampled_csc = pickle.load(infile)
            with open(RESIDUAL_CSC + '_user.pkl', 'rb') as infile:
                self.residual_csc = pickle.load(infile)
            print("Loaded sampled and residual matrices")
        except:
            self.original_dataset = self.dataset
            iteration = 0
            list_users = np.unique(self.dataset.nonzero()[0])
            csc_residual_dataset = sps.lil_matrix(self.dataset.shape)
            csc_sampled_dataset = sps.lil_matrix(self.dataset.shape)
            len_test = len(list_users)

            verbose = 3
            for u in list_users[:2]:
                iteration += 1
                user_profile = self.dataset[u,:]

                positive = user_profile > 3
                positive_items = positive.nonzero()[1]

                # Filtering on only positive ratings
                if verbose > 1:
                    print("\n\nProfile:\n",user_profile)
                    print("\n\nPositive\n",user_profile>3)
                    print("\n\nPositivi nonzero()[1]\n",positive.nonzero()[1])
                    print("\n\nFiltered\n",user_profile[0,positive.nonzero()[1]])

                if verbose > 0 and iteration%500 == 0:
                    print("\nIteration {} over {}".format(iteration, len_test))
                    #print("Profile:",user_profile)

                rated_pos_items = list(positive_items)
                if len(rated_pos_items) > k:
                    sampled_items = random.sample(rated_pos_items,k)
                    residual_items = list(set(rated_pos_items) - set(sampled_items))
                    if verbose > 1:
                        print("All items",rated_pos_items)
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
            with open(SAMPLED_CSC + '_user.pkl', 'wb') as outfile:
                pickle.dump(self.sampled_csc, outfile, pickle.HIGHEST_PROTOCOL)
            with open(RESIDUAL_CSC +  '_user.pkl', 'wb') as outfile:
                pickle.dump(self.residual_csc, outfile, pickle.HIGHEST_PROTOCOL)
            print("Saved sampled and residual matrices")

    def holdout(self, cut_perc = 0.30, verbose = 0):
        # try to load the matrices, otherwise compute and store them
        try:
            with open(SAMPLED_CSC + '.pkl', 'rb') as infile:
                self.sampled_csc = pickle.load(infile)
            with open(RESIDUAL_CSC + '.pkl', 'rb') as infile:
                self.residual_csc = pickle.load(infile)
        except:
            self.original_dataset = self.dataset
            
            csc_residual_dataset = sps.lil_matrix(self.dataset.shape)
            csc_sampled_dataset = sps.lil_matrix(self.dataset.shape)
            
            # Positive filtering
            positive_dataset = self.dataset > 3

            list_full_indexes = list(zip(self.dataset.nonzero()[0],self.dataset.nonzero()[1]))
            list_positive_indexes = list(zip(positive_dataset.nonzero()[0],positive_dataset.nonzero()[1]))

            if verbose > 0:
                print("Num full ind:{}, pos ind:{}".format(len(list_full_indexes), len(list_positive_indexes)))
                #print("NNZ full ds: {}, pos ds: {}".format(self.dataset.nnz, positive_dataset.nnz))
                # Qui ci starebbe bene una assert sulle lunghezze ....
                #print("Pos indexes:", list_positive_indexes) 
            n_cutted = int(len(list_positive_indexes) * cut_perc)

            selected_positive = random.sample(list_positive_indexes, n_cutted)
            residual_indexes = list(set(list_full_indexes) - set(selected_positive))

            for i,j in selected_positive:
                csc_sampled_dataset[i,j] = self.dataset[i,j]

            for i,j in residual_indexes:
                csc_residual_dataset[i,j] = self.dataset[i,j]

            if verbose > 3:
                print("Sampled ", csc_sampled_dataset)
                print("Residual ", csc_residual_dataset)

            self.sampled_csc = csc_sampled_dataset
            self.residual_csc = csc_residual_dataset

            with open(RESIDUAL_CSC + '.pkl', 'wb') as outfile:
                pickle.dump(csc_residual_dataset, outfile, pickle.HIGHEST_PROTOCOL)
            with open(SAMPLED_CSC + '.pkl', 'wb') as outfile:
                pickle.dump(csc_sampled_dataset, outfile, pickle.HIGHEST_PROTOCOL)
            print("Computed and stored residual and sampled matrices")
