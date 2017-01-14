from recSysLib.data_utils import load_sparse,save_sparse
from netflix_reader import NetflixReader
from recSysLib import slim

TO_COMPUTE_SLIM = False

# Get the URM

netflixReader = NetflixReader()
netflix_urm = netflixReader.urm

if TO_COMPUTE_SLIM:
    # gridSearch on l1_penalty, l2_penalty of Slim
    model = slim.Slim()
    model.fit(netflix_urm,verbose=1)
    weight_matrix = model.get_weight_matrix()
    save_sparse('../datasources/slim_W01',weight_matrix)
else:
    weight_matrix = load_sparse('../datasources/slim_W01.npz','csc')
    print(weight_matrix.shape)