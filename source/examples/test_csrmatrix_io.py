from recSysLib import slim
from recSysLib.data_utils import store_sparse_mat, load_sparse_mat, get_urm

folder_path = '../../datasources/'

netflix_urm = get_urm()

# Compute Weight Matrix
l1, l2 = 0.01, 0.01
model = slim.MultiThreadSLIM(l1_penalty=l1,l2_penalty=l2)
model.fit(netflix_urm)
weight_matrix = model.get_weight_matrix()

print(type(weight_matrix))
# Store Weight Matrix
store_sparse_mat(weight_matrix, folder_path + 'weight_matrix_test')

# Load Weight Matrix
weight_matrix_loaded = load_sparse_mat(folder_path + 'weight_matrix_test')

print("Stored Matrix: \n",weight_matrix)

print("Loaded Matrix: \n",weight_matrix_loaded)