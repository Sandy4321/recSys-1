import scipy.io as sio
import numpy as np

BASEFILE = "../datasets/Enriched_Netflix_Dataset/"

icm = sio.loadmat(BASEFILE + "./icm.mat")
icm_matrix = icm['icm']
icm_dictionary = icm['dictionary']
icm_stems = icm_dictionary['stems'][0][0]
icm_stemtypes = icm_dictionary['stemTypes'][0][0]

titles = sio.loadmat(BASEFILE + "./titles.mat")['titles']

urm = sio.loadmat(BASEFILE + "./urm.mat")['urm']


#Now let's try to aggregate the features

#The first group is composed by the actors
actor_features_names = ['ActorsLastNameFirstArray', 'ActorsLastNameFirstArray;DirectorsLastNameFirstArray']
actor_features = list()
for feature_name in actor_features_names:
    actor_features.extend(np.where(icm_stemtypes == feature_name)[0].tolist())
#you can now obtain the relevant icm with icm_matrix[actor_features]
