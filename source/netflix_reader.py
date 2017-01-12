import scipy.io as sio

BASEFILE = "../datasets/Enriched_Netflix_Dataset/"

icm = sio.loadmat(BASEFILE + "./icm.mat")
icm_matrix = icm['icm']
icm_dictionary = icm['dictionary']
icm_stems = icm_dictionary['stems'][0][0]
icm_stemtypes = icm_dictionary['stemTypes'][0][0]

titles = sio.loadmat(BASEFILE + "./titles.mat")['titles']

urm = sio.loadmat(BASEFILE + "./urm.mat")['urm']
