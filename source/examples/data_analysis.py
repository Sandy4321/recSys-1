import scipy as scp
import numpy as np

from recSysLib import netflix_reader
from recSysLib.data_utils import get_urm

import os, matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import brewer2mpl

urm = get_urm()

netflix_reader = netflix_reader.NetflixReader()

icm = netflix_reader._icm_reduced_matrix

print("SHAPES urm: {}, icm: {}".format(urm.shape, icm.shape))

print("\n===URM general Sparsity===")
print("Number of non zero elements", urm.getnnz())
print("Number of non zero elements over max",
      urm.getnnz()/(urm.shape[0]*urm.shape[1]))

print("\n===ICM general Sparsity===")
print("Number of non zero elements", icm.getnnz())
print("Number of non zero elements over max",
      icm.getnnz()/(icm.shape[0]*icm.shape[1]))

print("\n===URM Sparsity along 1_dimension===")
print("Number of ratings by users", urm.getnnz(axis = 1),
      np.min(urm.getnnz(axis=1)), np.max(urm.getnnz(axis=1)))
print("Number of ratings by items", urm.getnnz(axis = 0),
      np.min(urm.getnnz(axis=0)), np.max(urm.getnnz(axis=0)))
ratings_by_users = urm.getnnz(axis = 1)
ratings_by_items = urm.getnnz(axis = 0)

print("\n===ICM Sparsity along 1_dimension===")
print("Number of features by items", icm.getnnz(axis = 1),
     np.min(icm.getnnz(axis=1)), np.max(icm.getnnz(axis=1)))
print("Number of features by features", icm.getnnz(axis = 0),
     np.min(icm.getnnz(axis=0)), np.max(icm.getnnz(axis=0)))
features_by_items = icm.getnnz(axis = 1)
features_by_feature = icm.getnnz(axis = 0)

def _plot_distributions(array, title, label, path):
    plt.hist(array, bins=10, )
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.savefig(path)
    plt.close()
    #sns.distplot(array)

_plot_distributions(ratings_by_users, title='Rating frequency per user',
                    label='number of ratings', path='./rates_users.png')
_plot_distributions(ratings_by_items, title='Rating frequency per item',
                    label='number of ratings', path='./rates_items.png')
_plot_distributions(features_by_feature, title='Feature frequency per feature',
                    label='number of features', path='./features_feature.png')
_plot_distributions(features_by_items, title='Feature frequency per item',
                    label='number of features', path='./features_items.png')
