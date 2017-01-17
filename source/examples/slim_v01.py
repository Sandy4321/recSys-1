from recSysLib.data_split import holdout
from recSysLib.data_utils import get_urm
from recSysLib.metrics import precision, recall
from recSysLib import slim

import numpy as np

TO_COMPUTE_SLIM = True

# Get the URM
netflix_urm = get_urm()

if TO_COMPUTE_SLIM:

    # Split in train and validation by holdout
    sub_train , sub_validation = holdout(netflix_urm, perc=0.85, clean_test=True)

    indexes = sub_validation.nonzero()
    print("Indexes:",indexes)
    print("Indexes[0]",indexes[0])
    print("Indexes[0]unique",np.unique(indexes[0]))

    users_validation = np.unique(indexes[0])

    iteration = 0
    for i in range(-5,2,2):
        iteration += 1
        l1 = 10**i
        l2 = 0
        #l2 = numbernp.float32(random.uniform(0, 10.0))

        # Compute the model with the established parameters
        model = slim.MultiThreadSLIM(l1_penalty=l1,l2_penalty=l2)
        model.fit(sub_train)
        weight_matrix = model.get_weight_matrix()


        # Evaluation
        n_eval = 0
        metric_ = 0.0
        metric_1 = 0.0
        for user_to_test in users_validation:
            print("User: {} \n Item indices {}".format(user_to_test,sub_validation[user_to_test].indices))
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
