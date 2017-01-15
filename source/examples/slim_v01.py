from recSysLib.data_utils import load_sparse,save_sparse
from netflix_reader import NetflixReader
from recSysLib import slim
import numpy as np
import joblib

TO_COMPUTE_SLIM = False

# Get the URM

netflixReader = NetflixReader()
netflix_urm = netflixReader.urm

print("\n===URM User Rating Matrix===")
print("\nData Len: {}".format(len(netflix_urm.data)))
print("Data shape: {}".format(netflix_urm.shape))
print("Zero elements: {}".format((netflix_urm.shape[0]*netflix_urm.shape[1]) - netflix_urm.getnnz()))

print("\nNon zero elements by USERS")
print("Non zero elements all rows: {}".format(netflix_urm.getnnz(axis=1)))
print("Non zero elements all rows len: {}".format(len(netflix_urm.getnnz(axis=1))))
print("Non zero elements all rows sum: {}".format(np.sum(netflix_urm.getnnz(axis=1))))
print("Non zero elements all rows avg: {}".format(np.sum(netflix_urm.getnnz(axis=1))/len(netflix_urm.getnnz(axis=1))))

print("\nNon zero elements by ITEMS")
print("Non zero elements all cols: {}".format(netflix_urm.getnnz(axis=0)))
print("Non zero elements all cols len: {}".format(len(netflix_urm.getnnz(axis=0))))
print("Non zero elements all cols sum: {}".format(np.sum(netflix_urm.getnnz(axis=0))))
print("Non zero elements all cols avg: {}".format(np.sum(netflix_urm.getnnz(axis=0))/len(netflix_urm.getnnz(axis=0))))


if TO_COMPUTE_SLIM:
    # gridSearch on l1_penalty, l2_penalty of Slim... qua ci sarebbe da ragionarci un po'.... TODO
    model = slim.Slim()
    model.fit(netflix_urm,verbose=0)
    weight_matrix = model.get_weight_matrix()
    save_sparse('../datasources/slim_W01.npz',weight_matrix.toarray())
else:
    weight_matrix = load_sparse('../datasources/slim_W01.npz','csc')

print("\n===SLIM ITEM-WEIGHT-SIMILARITY MATRIX===")
print("Shape: {}".format(weight_matrix.shape))
print("Non zero elements: {}".format(weight_matrix.getnnz()))
print("Non zero elements all cols len: {}".format(len(weight_matrix.getnnz(axis=0))))
print("Non zero elements all cols sum: {}".format(np.sum(weight_matrix.getnnz(axis=0))))
print("Non zero elements all cols avg: {}".format(np.sum(weight_matrix.getnnz(axis=0))/len(weight_matrix.getnnz(axis=0))))

