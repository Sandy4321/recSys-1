#Author: Stefano Cereda
import scipy.io as sio
import numpy as np
from difflib import SequenceMatcher
import pickle
import multiprocessing
import math


BASEFILE = "../../datasets/Enriched_Netflix_Dataset/"
ICM_DICTIONARY_FILE = "../../datasources/netflix/icm_dict.pkl"
PRODUCTS_MATRIX_DIR = "../../datasources/netflix/f_prod_mat/"
NUM_PROD_MAT = 21


class NetflixReader:
    def __init__(self):
        self._urm = sio.loadmat(BASEFILE + "./urm.mat")['urm']
        self.numItems = self._urm.shape[1]
        
        self._icm = sio.loadmat(BASEFILE + "./icm.mat")
        self._icm_matrix = self._icm['icm']
        self._icm_dictionary = self._icm['dictionary']
        self._icm_stems = self._icm_dictionary['stems'][0][0]
        self._icm_stemtypes = self._icm_dictionary['stemTypes'][0][0]
        self._titles = sio.loadmat(BASEFILE + "./titles.mat")['titles']
        self._urm = sio.loadmat(BASEFILE + "./urm.mat")['urm']
        
        #Now let's try to aggregate the features
        #The first group is composed by the actors
        actor_features_names = ['ActorsLastNameFirstArray', 'ActorsLastNameFirstArray;DirectorsLastNameFirstArray']
        self._actor_features = list()
        for feature_name in actor_features_names:
            self._actor_features.extend(np.where(self._icm_stemtypes == feature_name)[0].tolist())

        #the second group of features is the country of production
        county_features_names = ['ChannelName;CountryOfOrigin', 'CountryOfOrigin']
        self._country_features = list()
        for feature_name in county_features_names:
            self._country_features.extend(np.where(self._icm_stemtypes == feature_name)[0].tolist())
        
        #the third group of features is the director, with just one name 
        self._director_features = np.where(self._icm_stemtypes == 'DirectorsLastNameFirstArray')[0].tolist()

        #we now have the genres.
        #TODO: Bear in mind that some of these features are extracted from tags/stems so it could be worth separating them
        genres_features_names = ['GenresArray', 'GenresArray;KeywordsArray', 'GenresArray;KeywordsArray;TitleFull', 'GenresArray;TitleFull']
        self._genres_features = list()
        for feature_name in genres_features_names:
            self._genres_features.extend(np.where(self._icm_stemtypes == feature_name)[0].tolist())

        #skip the tags (for now) TODO and go to miniseries
        self._miniseries_features = np.where(self._icm_stemtypes == 'ShowType')[0].tolist()

        #Skip the title as we have a separate file fo them and do the year
        self._years_features = np.where(self._icm_stemtypes == 'Year')[0].tolist()
        
        
        
        
        #try to load the icm dictionary
        try:
            with open(ICM_DICTIONARY_FILE, 'rb') as f:
                self._icm_dict = pickle.load(f)
        except:
            print("Building icm dictionary")
            self._build_feature_dictionary()
            with open(ICM_DICTIONARY_FILE, 'wb') as f:
                pickle.dump(self._icm_dict, f, pickle.HIGHEST_PROTOCOL)
        
        #try to load the matrix of products
        try:
            print("TOGLIMI self._load_prod_mat()")
            #self._load_prod_mat()
        except:
            print("Building features products matrix")
            self._build_products_matrix()
            self._store_prod_mat()
            self._load_prod_mat()
    
    def test_sorting(self):
        self._sort_genres_by_pop()
        print("Sorting by genres: completed!")
        self._sort_actor_by_pop()
        print("Sorting by actor: completed!")
        self._sort_country_by_pop()
        print("Sorting by country: completed!")
        self._sort_director_by_pop()
        print("Sorting by director: completed!")
        #print("SORTING: \ngenres:{}, \nactor:{}, \ncountry:{}, \ndirector:{}".format(self._genres_features_sorted,self._actor_features_sorted,self._country_features_sorted,self._director_features_sorted))

    def test_cut(self):
        for k in range(1,10):
            print("\n\n===CUTTING {}===".format(k))
            print("\nGenres:")
            self._cut_genres_by_pop(k)
            print("\nActor:")
            self._cut_actor_by_pop(k)
            print("\nCountry:")
            self._cut_country_by_pop(k)
            print("\nDirector:")
            self._cut_director_by_pop(k)

    def _cut_genres_by_pop(self, k):
        list_cutted_features = []
        for couple in self._genres_features_sorted:
            ind, freq = couple
            if freq < k:
                print("Only {} over {} elements have been selected".format(len(list_cutted_features),len(self._genres_features_sorted)))
                return list_cutted_features
            list_cutted_features.append(ind)

    def _cut_actor_by_pop(self, k):
        list_cutted_features = []
        for couple in self._actor_features_sorted:
            ind, freq = couple
            if freq < k:
                print("Only {} over {} elements have been selected".format(len(list_cutted_features),len(self._actor_features_sorted)))
                return list_cutted_features
            list_cutted_features.append(ind)

    def _cut_country_by_pop(self, k):
        list_cutted_features = []
        for couple in self._country_features_sorted:
            ind, freq = couple
            if freq < k:
                print("Only {} over {} elements have been selected".format(len(list_cutted_features),len(self._country_features_sorted)))
                return list_cutted_features
            list_cutted_features.append(ind)

    def _cut_director_by_pop(self, k):
        list_cutted_features = []
        for couple in self._director_features_sorted:
            ind, freq = couple
            if freq < k:
                print("Only {} over {} elements have been selected".format(len(list_cutted_features),len(self._director_features_sorted)))
                return list_cutted_features
            list_cutted_features.append(ind)

    def _sort_genres_by_pop(self):
        list_tuples = []
        genres_indexes = self._genres_features
        #print("Genres indexes",genres_indexes)
        for genres_index in genres_indexes:
            #print("Index: {} , len: {}".format(genres_index,len(self._icm_matrix[genres_index,:].nonzero()[1])))
            list_tuples.append((genres_index,len(self._icm_matrix[genres_index,:].nonzero()[1])))

        #print("original:",list_tuples)
        list_tuples.sort(key=lambda tup: tup[1],reverse=True)
        self._genres_features_sorted = list_tuples
    
    def _sort_actor_by_pop(self):
        list_tuples = []
        actor_indexes = self._actor_features
        for actor_index in actor_indexes:
            list_tuples.append((actor_index,len(self._icm_matrix[actor_index,:].nonzero()[1])))

        list_tuples.sort(key=lambda tup: tup[1],reverse=True)
        self._actor_features_sorted = list_tuples

    def _sort_country_by_pop(self):
        list_tuples = []
        country_indexes = self._country_features
        for country_index in country_indexes:
            list_tuples.append((country_index,len(self._icm_matrix[country_index,:].nonzero()[1])))

        list_tuples.sort(key=lambda tup: tup[1],reverse=True)
        self._country_features_sorted = list_tuples

    def _sort_director_by_pop(self):
        list_tuples = []
        director_indexes = self._director_features
        for director_index in director_indexes:
            list_tuples.append((director_index,len(self._icm_matrix[director_index,:].nonzero()[1])))

        list_tuples.sort(key=lambda tup: tup[1],reverse=True)
        self._director_features_sorted = list_tuples

    def _store_prod_mat(self):
        for i in range(NUM_PROD_MAT):
            m = self._prod_mat[i]
            np.save(PRODUCTS_MATRIX_DIR+str(i)+".npy", m)

    def _load_prod_mat(self):
        self._prod_mat = list()
        for i in range(NUM_PROD_MAT):
            self._prod_mat.append(np.load(PRODUCTS_MATRIX_DIR+str(i)+".npy"))


    ###CONVERT THE MATRIX TO A DICTIONARY###
    def _build_feature_dictionary(self):
        self._icm_dict = {}
        for itemId in range(0, self._icm_matrix.shape[1]):
            self._icm_dict[itemId] = list()
            self._icm_dict[itemId].append(self._icm_matrix[self._actor_features, itemId].astype(bool))
            self._icm_dict[itemId].append(self._icm_matrix[self._country_features, itemId].astype(bool))
            self._icm_dict[itemId].append(self._icm_matrix[self._director_features, itemId].astype(bool))
            self._icm_dict[itemId].append(self._icm_matrix[self._genres_features, itemId].astype(bool))
            self._icm_dict[itemId].append(self._icm_matrix[self._miniseries_features, itemId].astype(bool).toarray()[0][0])
            year_feat = self._icm_matrix[self._years_features, itemId].astype(bool).indices[0]
            self._icm_dict[itemId].append(int(self._icm_stems[self._years_features[year_feat]][0][0]))
            self._icm_dict[itemId].append(self._titles[itemId])
            
            
    ###COMPUTE A MATRIX OF FEATURES PRODUCTS###
    def _build_products_matrix(self):
        num_items_each = math.ceil(self.numItems/NUM_PROD_MAT)

        idx = list()
        for i in range(NUM_PROD_MAT):
            start = num_items_each * i
            end = min(start + num_items_each, self.numItems)
            idx.append((start,end))
        
        pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
        res = pool.map(_help_compute_matrix, ((self, s, e) for (s,e) in idx))
        
        self._prod_mat = res
            

    
    #compute f_{i;actors} * f_{j;actors} by counting the number of common actors
    #TODO: is better to use the number of common actors or normalize it against the number of featured actors?
    def _similarity_actors(self, movieId1, movieId2):
        actors_1 = self._icm_dict[movieId1][0]
        actors_2 = self._icm_dict[movieId2][0]

        num_act = actors_1.sum() + actors_2.sum()
        num_commons = (actors_1.multiply(actors_2)).sum()
        return 2*num_commons/max(1,num_act)
        
        
    #we can also use the number of actors to compute teh similarity between the cast dimension
    def _similarity_cast_dimension(self, movieId1, movieId2):
        actors_1 = self._icm_dict[movieId1][0]
        actors_2 = self._icm_dict[movieId2][0]

        num_act_1 = actors_1.sum()
        num_act_2 = actors_2.sum()
        
        return 1/(1+abs(num_act_1 - num_act_2))


    #compute the similarity of the origin country by checking if they are the same
    def _similarity_countries(self, movieId1, movieId2):
        country_1 = self._icm_dict[movieId1][1]
        country_2 = self._icm_dict[movieId2][1]

        return (country_1.multiply(country_2)).sum()


    #check if the director is the same
    def _similarity_directors(self, movieId1, movieId2):
        dir_1 = self._icm_dict[movieId1][2]
        dir_2 = self._icm_dict[movieId2][2]
    
        num_directors = dir_1.sum() + dir_2.sum()
        num_commons = (dir_1.multiply(dir_2)).sum()
        
        if num_directors == 0:
            return 0.1
        
        return 2* num_commons / max(1,num_directors)


    #The similarity between genres is given by the number of common genres
    #TODO: like for the actors, would a percentage be better?
    def _similarity_genres(self, movieId1, movieId2):
        genres_1 = self._icm_dict[movieId1][3]
        genres_2 = self._icm_dict[movieId2][3]

        num_genres = genres_1.sum() + genres_2.sum()
        num_commons = (genres_1.multiply(genres_2)).sum()
        
        return 2* num_commons / max(1,num_genres)
        
        
    #As for the actors, we can use the "complexity" of the genres
    def _similarity_num_genres(self, movieId1, movieId2):
        genres_1 = self._icm_dict[movieId1][3]
        genres_2 = self._icm_dict[movieId2][3]

        num_genres_1 = genres_1.sum()
        num_genres_2 = genres_2.sum()
        
        return 1/(1+abs(num_genres_1-num_genres_2))
        
        
    #check if the two items are both miniseries OR if they are both NOT miniseries
    def _similarity_miniseries(self, movieId1, movieId2):
        type_1 = self._icm_dict[movieId1][4]
        type_2 = self._icm_dict[movieId2][4]

        return (type_1 == type_2).sum()


    #The similarity of two years is the inverse of their difference
    #TODO: maybe try a more refined approach
    def _similarity_years(self, movieId1, movieId2):
        year_1 = self._icm_dict[movieId1][5]
        year_2 = self._icm_dict[movieId2][5]
        
        return 1/(1+abs(year_1-year_2))
        

    #The similarity of two titles is the number of common words
    def _similarity_titles(self, movieId1, movieId2):
        title_1 = self._icm_dict[movieId1][6]
        title_2 = self._icm_dict[movieId1][6]
        return SequenceMatcher(lambda x: x in " \t", title_1.lower(), title_2.lower()).ratio() #skip the blanks
    
    
    #return all the "feature products" between two items
    def _similarity(self, movieId1, movieId2):
        return (self._similarity_actors(movieId1, movieId2),
                self._similarity_cast_dimension(movieId1, movieId2),
                self._similarity_countries(movieId1, movieId2),
                self._similarity_directors(movieId1, movieId2),
                self._similarity_genres(movieId1, movieId2),
                self._similarity_num_genres(movieId1, movieId2),
                self._similarity_miniseries(movieId1, movieId2),
                self._similarity_years(movieId1, movieId2),
                self._similarity_titles(movieId1, movieId2))
        
    
    ###PUBLIC METHOD TO BE CALLED TO OBTAIN THE FEATURES PRODUCTS BETWEEN TWO ITEMS###
    def get_similarity(self, i, j):
        if i == j:
            raise ValueError('Asking similarity of an item with itself')
        if i > j:
            t = i
            i = j
            j = i
        
        #we have stored the data on separate matrix
        num_items_each = math.ceil(self.numItems/NUM_PROD_MAT)
        num_m = math.floor(i / num_items_each)
        i_o = i % num_items_each
        
        return np.array(self._prod_mat[num_m][i_o,j], dtype=np.float32)
    
    def get_similarity_tuple(self, ij):
        return self.get_similarity(ij[0], ij[1])
            

###USED TO PARALLELY COMPUTE THE SIMILARITY MATRIX###
#i1 end IS NOT COMPUTED
def _help_compute_matrix(x):
    (reader, i1start, i1end) = x
    matrix = np.zeros((i1end-i1start, reader.numItems), dtype=object)
    
    for movieId1 in range(i1start, i1end):
        print(movieId1)
        for movieId2 in range(movieId1+1, reader.numItems):
            matrix[movieId1-i1start, movieId2] = reader._similarity(movieId1, movieId2)
            
    return matrix



if __name__ == '__main__':
    a = NetflixReader()
    print("loaded")
    
    import time
    import multiprocessing

    a.test_sorting()
    a.test_cut()

    tic = time.time()
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
#    res=pool.map(a.get_similarity_tuple, ((i,j) for i in range(1,100) for j in range(i+1,100)))
    toc = time.time()
    print("The parallel version took %f seconds" %(toc-tic))
    
    tic = time.time()
    for i in range(0,a.numItems):
        for j in range(i+1,a.numItems):
            a.get_similarity(i,j)
    toc = time.time()
    print("The serial version took %f seconds" %(toc-tic))


