from recSysLib.data_utils import load_sparse,save_sparse
from netflix_reader import NetflixReader
from recSysLib import slim

# Get the URM

netflixReader = NetflixReader()
netflix_urm = netflixReader.urm


model = slim.Slim()
model.fit(netflix_urm,verbose=1)
weight_matrix = model.get_weight_matrix()

print("Slim-W similarity items: class type: {}, shape: {}".format(type(weight_matrix),weight_matrix.shape))

save_sparse('../datasources/slim_W01',weight_matrix)