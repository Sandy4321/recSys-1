from recSysLib.dataset_partition import DataPartition
from recSysLib.data_utils import get_urm
percentage_train_row = 0.9
percentage_train_col = 0.9

urm = get_urm()

urm_partition = DataPartition(urm)
urm_partition.split_train_test(train_perc_col=percentage_train_col, train_perc_row=percentage_train_row,verbose=1)

print(urm_partition.toString())

train_matrix = urm_partition.get_train_matrix()
test_matrix = urm_partition.get_test_matrix()

print("Train shape: {} , \nTest shape: {}".format(train_matrix.shape,test_matrix.shape))