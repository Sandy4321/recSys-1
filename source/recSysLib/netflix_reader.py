#Author: Stefano Cereda
import scipy.io as sio
import numpy as np
from difflib import SequenceMatcher
import pickle

BASEFILE = "../../datasets/Enriched_Netflix_Dataset/"
DICTIONARY_FILE = "../../datasources/netflix/dict.pkl"

class NetflixReader:
    def __init__(self):
        self._urm = sio.loadmat(BASEFILE + "./urm.mat")['urm']
        
        #try to load the dictionary
        try:
            with open(DICTIONARY_FILE, 'rb') as f:
                self._dict = pickle.load(f)
        except:
            print("Building dictionary")
        
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
            
            self._build_feature_dictionary()
            with open(DICTIONARY_FILE, 'wb') as f:
                pickle.dump(self._dict, f, pickle.HIGHEST_PROTOCOL)
    
    
    
    ###CONVERT THE MATRIX TO A DICTIONARY###
    def _build_feature_dictionary(self):
        self._dict = {}
        for itemId in range(0, self._icm_matrix.shape[1]):
            self._dict[itemId] = list()
            self._dict[itemId].append(self._icm_matrix[self._actor_features, itemId].astype(bool))
            self._dict[itemId].append(self._icm_matrix[self._country_features, itemId].astype(bool))
            self._dict[itemId].append(self._icm_matrix[self._director_features, itemId].astype(bool))
            self._dict[itemId].append(self._icm_matrix[self._genres_features, itemId].astype(bool))
            self._dict[itemId].append(self._icm_matrix[self._miniseries_features, itemId].astype(bool).toarray()[0][0])
            year_feat = self._icm_matrix[self._years_features, itemId].astype(bool).indices[0]
            self._dict[itemId].append(int(self._icm_stems[self._years_features[year_feat]][0][0]))
            self._dict[itemId].append(self._titles[itemId])
            
            
    
    #compute f_{i;actors} * f_{j;actors} by counting the number of common actors
    #TODO: is better to use the number of common actors or normalize it against the number of featured actors?
    def _similarity_actors(self, movieId1, movieId2):
        actors_1 = self._dict[movieId1][0]
        actors_2 = self._dict[movieId2][0]

        num_act = actors_1.sum() + actors_2.sum()
        num_commons = (actors_1.multiply(actors_2)).sum()
        return 2*num_commons/num_act
        
        
    #we can also use the number of actors to compute teh similarity between the cast dimension
    def _similarity_cast_dimension(self, movieId1, movieId2):
        actors_1 = self._dict[movieId1][0]
        actors_2 = self._dict[movieId2][0]

        num_act_1 = actors_1.sum()
        num_act_2 = actors_2.sum()
        
        return 1/(1+abs(num_act_1 - num_act_2))


    #compute the similarity of the origin country by checking if they are the same
    def _similarity_countries(self, movieId1, movieId2):
        country_1 = self._dict[movieId1][1]
        country_2 = self._dict[movieId2][1]

        return (country_1.multiply(country_2)).sum()


    #check if the director is the same
    def _similarity_directors(self, movieId1, movieId2):
        dir_1 = self._dict[movieId1][2]
        dir_2 = self._dict[movieId2][2]
    
        num_directors = dir_1.sum() + dir_2.sum()
        num_commons = (dir_1.multiply(dir_2)).sum()
        
        if num_directors == 0:
            return 0.1
        
        return 2* num_commons / num_directors


    #The similarity between genres is given by the number of common genres
    #TODO: like for the actors, would a percentage be better?
    def _similarity_genres(self, movieId1, movieId2):
        genres_1 = self._dict[movieId1][3]
        genres_2 = self._dict[movieId2][3]

        num_genres = genres_1.sum() + genres_2.sum()
        num_commons = (genres_1.multiply(genres_2)).sum()
        
        return 2* num_commons / num_genres
        
        
    #As for the actors, we can use the "complexity" of the genres
    def _similarity_num_genres(self, movieId1, movieId2):
        genres_1 = self._dict[movieId1][3]
        genres_2 = self._dict[movieId2][3]

        num_genres_1 = genres_1.sum()
        num_genres_2 = genres_2.sum()
        
        return 1/(1+abs(num_genres_1-num_genres_2))
        
        
    #check if the two items are both miniseries OR if they are both NOT miniseries
    def _similarity_miniseries(self, movieId1, movieId2):
        type_1 = self._dict[movieId1][4]
        type_2 = self._dict[movieId2][4]

        return (type_1 == type_2).sum()


    #The similarity of two years is the inverse of their difference
    #TODO: maybe try a more refined approach
    def _similarity_years(self, movieId1, movieId2):
        year_1 = self._dict[movieId1][5]
        year_2 = self._dict[movieId2][5]
        
        return 1/(1+abs(year_1-year_2))
        

    #The similarity of two titles is the number of common words
    def _similarity_titles(self, movieId1, movieId2):
        title_1 = self._dict[movieId1][6]
        title_2 = self._dict[movieId1][6]
        return SequenceMatcher(lambda x: x in " \t", title_1.lower(), title_2.lower()).ratio() #skip the blanks
    
    
    #return all the "feature products" between two items
    def similarity(self, movieId1, movieId2):
        return (self._similarity_actors(movieId1, movieId2),
                self._similarity_cast_dimension(movieId1, movieId2),
                self._similarity_countries(movieId1, movieId2),
                self._similarity_directors(movieId1, movieId2),
                self._similarity_genres(movieId1, movieId2),
                self._similarity_num_genres(movieId1, movieId2),
                self._similarity_miniseries(movieId1, movieId2),
                self._similarity_years(movieId1, movieId2),
                self._similarity_titles(movieId1, movieId2))
    
    
    def similarity_tuple(self, ij):
        return self.similarity(ij[0], ij[1])
    
    


    
    #return the rating of an user for a movie
    #WARNING: 0 means the rating is missing
    def get_rating(self, userId, movieId):
        return self.urm[userId, movieId]
        
    
    
if __name__ == '__main__':
    a = NetflixReader()
    print("loaded")
    
    def _compute(ij):
        i = ij[0]
        j = ij[1]
        return (i,j,a.similarity(i,j))
    
    import time
    import multiprocessing

    tic = time.time()
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    res=pool.map(_compute, ((i,j) for i in range(1,100) for j in range(i+1,100)))
    toc = time.time()
    print("The parallel version took %f seconds" %(toc-tic))
    
    tic = time.time()
    for i in range(1,100):
        for j in range(i+1,100):
            _compute((i,j))
    toc = time.time()
    print("The serial version took %f seconds" %(toc-tic))


