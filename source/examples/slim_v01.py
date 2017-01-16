from recSysLib.netflix_reader import NetflixReader
from recSysLib.data_split import holdout
from recSysLib import slim
import numpy as np

TO_COMPUTE_SLIM = True

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

max_nnz = 0
if TO_COMPUTE_SLIM:
    sub_train , sub_validation = holdout(netflix_urm, perc=0.9, clean_test=True)

    for l1 in [0.001 , 0.01 , 0.05 , 0.1 , 0.5]:
        for l2 in [0.001 , 0.01 , 0.05 , 0.1 , 0.5]:
            model = slim.MultiThreadSLIM(l1_penalty=l1,l2_penalty=l2)
            model.fit(sub_train)
            weight_matrix = model.get_weight_matrix()
            if weight_matrix.getnnz() > max_nnz:
                max_nnz = weight_matrix.getnnz()
                np.save('../datasources/slim_W01.npz',weight_matrix)
                print("l1: {} , l2: {} , len: {}".format(l1,l2,weight_matrix.getnnz()))
else:
    weight_matrix = np.load('../../datasources/slim_W01.npz','csc')

print("\n===SLIM ITEM-WEIGHT-SIMILARITY MATRIX===")
print("Shape: {}".format(weight_matrix.shape))
print("Non zero elements: {}".format(weight_matrix.getnnz()))
print("Non zero elements all cols len: {}".format(len(weight_matrix.getnnz(axis=0))))
print("Non zero elements all cols sum: {}".format(np.sum(weight_matrix.getnnz(axis=0))))
print("Non zero elements all cols avg: {}".format(np.sum(weight_matrix.getnnz(axis=0))/len(weight_matrix.getnnz(axis=0))))

