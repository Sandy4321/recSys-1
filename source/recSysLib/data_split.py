import numpy as np
import random
import scipy.sparse as sps
from sklearn.model_selection import train_test_split

def holdout(data, perc=0.99, seed=1234, clean_test=True):
    pos_r_inds = data > 3
    #print("LEN INDPTR: ",len(pos_r_inds.indptr))
    #print(pos_r_inds.nonzero())

    #print("LEN nnzer[0],[1]",len(pos_r_inds.nonzero()[0]),len(pos_r_inds.nonzero()[1]))
     
    tmp_l = len(pos_r_inds.nonzero()[0])
    tmp = tmp_l * (1 - perc) 
    validation_indexeszipped = random.sample(list(zip(pos_r_inds.nonzero()[0],pos_r_inds.nonzero()[1])), int(tmp))
    row_samples, col_samples = zip(*validation_indexeszipped) 

    #val_r = data[row_samples,col_samples]

    #print("Data shape",data.shape)
    new_mat_lil = sps.lil_matrix(data.shape, dtype = data.dtype)
    
    #print("PRE FOR")
    for point in validation_indexeszipped:
        #print("p[0][1]",point[0],point[1])
        new_mat_lil[point[0],point[1]] = data[point[0],point[1]]
    #print("MIDDLE")
    
    new_mat = new_mat_lil.tocsr()
    data[row_samples,col_samples] = 0
    return data, new_mat
