import scipy.io as sio
import numpy as np
from difflib import SequenceMatcher

BASEFILE = "../datasets/Enriched_Netflix_Dataset/"

class NetflixReader:
    def __init__(self):
        self.icm = sio.loadmat(BASEFILE + "./icm.mat")
        self.icm_matrix = self.icm['icm']
        self.icm_dictionary = self.icm['dictionary']
        self.icm_stems = self.icm_dictionary['stems'][0][0]
        self.icm_stemtypes = self.icm_dictionary['stemTypes'][0][0]
        self.titles = sio.loadmat(BASEFILE + "./titles.mat")['titles']
        self.urm = sio.loadmat(BASEFILE + "./urm.mat")['urm']

        #Now let's try to aggregate the features
        #The first group is composed by the actors
        actor_features_names = ['ActorsLastNameFirstArray', 'ActorsLastNameFirstArray;DirectorsLastNameFirstArray']
        self.actor_features = list()
        for feature_name in actor_features_names:
            self.actor_features.extend(np.where(self.icm_stemtypes == feature_name)[0].tolist())

        #the second group of features is the country of production
        county_features_names = ['ChannelName;CountryOfOrigin', 'CountryOfOrigin']
        self.country_features = list()
        for feature_name in county_features_names:
            self.country_features.extend(np.where(self.icm_stemtypes == feature_name)[0].tolist())
           
        #the third group of features is the director, with just one name 
        self.director_features = np.where(self.icm_stemtypes == 'DirectorsLastNameFirstArray')[0].tolist()

        #we now have the genres.
        #TODO: Bear in mind that some of these features are extracted from tags/stems so it could be worth separating them
        genres_features_names = ['GenresArray', 'GenresArray;KeywordsArray', 'GenresArray;KeywordsArray;TitleFull', 'GenresArray;TitleFull']
        self.genres_features = list()
        for feature_name in genres_features_names:
            self.genres_features.extend(np.where(self.icm_stemtypes == feature_name)[0].tolist())

        #skip the tags (for now) TODO and go to miniseries
        self.miniseries_features = np.where(self.icm_stemtypes == 'ShowType')[0].tolist()

        #Skip the title as we have a separate file fo them and do the year
        self.years_features = np.where(self.icm_stemtypes == 'Year')[0].tolist()
    
    
    
        
    #compute f_{i;actors} * f_{j;actors} by counting the number of common actors
    #TODO: is better to use the number of common actors or normalize it against the number of featured actors?
    def _similarity_actors(self, movieId1, movieId2):
        actors_1 = self.icm_matrix[self.actor_features, movieId1].astype(bool)
        actors_2 = self.icm_matrix[self.actor_features, movieId2].astype(bool)

        num_act = np.sum(actors_1) + np.sum(actors_2)
        num_commons = np.sum(actors_1.multiply(actors_2))
        return 2*num_commons/num_act
        
        
    #we can also use the number of actors to compute teh similarity between the cast dimension
    def _similarity_cast_dimension(self, movieId1, movieId2):
        actors_1 = self.icm_matrix[self.actor_features, movieId1].astype(bool)
        actors_2 = self.icm_matrix[self.actor_features, movieId2].astype(bool)

        num_act_1 = np.sum(actors_1)
        num_act_2 = np.sum(actors_2)
        
        return 1/(1+abs(num_act_1 - num_act_2))


    #compute the similarity of the origin country by checking if they are the same
    def _similarity_countries(self, movieId1, movieId2):
        country_1 = self.icm_matrix[self.country_features, movieId1].astype(bool)
        country_2 = self.icm_matrix[self.country_features, movieId2].astype(bool)

        return np.sum(country_1.multiply(country_2))


    #check if the director is the same
    def _similarity_directors(self, movieId1, movieId2):
        dir_1 = self.icm_matrix[self.director_features, movieId1].astype(bool)
        dir_2 = self.icm_matrix[self.director_features, movieId2].astype(bool)

        return np.sum(dir_1.multiply(dir_2))


    #The similarity between genres is given by the number of common genres
    #TODO: like for the actors, would a percentage be better?
    def _similarity_genres(self, movieId1, movieId2):
        genres_1 = self.icm_matrix[self.genres_features, movieId1].astype(bool)
        genres_2 = self.icm_matrix[self.genres_features, movieId2].astype(bool)

        #num_genres = np.sum(genres_1) + np.sum(genres_2)
        num_commons = np.sum(genres_1.multiply(genres_2))
        
        return num_commons
        
        
    #As for the actors, we can use the "complexity" of the genres
    def _similarity_num_genres(self, movieId1, movieId2):
        genres_1 = self.icm_matrix[self.genres_features, movieId1].astype(bool)
        genres_2 = self.icm_matrix[self.genres_features, movieId2].astype(bool)

        num_genres_1 = np.sum(genres_1)
        num_genres_2 = np.sum(genres_2)
        
        return 1/(1+abs(num_genres_1-num_genres_2))
        
        
    #check if the two items are both miniseries OR if they are both NOT miniseries
    def _similarity_miniseries(self, movieId1, movieId2):
        type_1 = self.icm_matrix[self.miniseries_features, movieId1].astype(bool).toarray()[0][0]
        type_2 = self.icm_matrix[self.miniseries_features, movieId2].astype(bool).toarray()[0][0]

        return np.sum(type_1 == type_2)


    #The similarity of two years is the inverse of their difference
    #TODO: maybe try a more refined approach
    def _similarity_years(self, movieId1, movieId2):
        year_feat_1 = self.icm_matrix[self.years_features, movieId1].astype(bool).indices[0]
        year_feat_2 = self.icm_matrix[self.years_features, movieId2].astype(bool).indices[0]
        
        year_1 = int(self.icm_stems[self.years_features[year_feat_1]][0][0])
        year_2 = int(self.icm_stems[self.years_features[year_feat_2]][0][0])

        return 1/(1+abs(year_1-year_2))
        

    #The similarity of two titles is the number of common words
    def _similarity_titles(self, movieId1, movieId2):
        title_1 = self.titles[movieId1]
        title_2 = self.titles[movieId2]
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
    res=pool.map(_compute, ((i,j) for i in range(1,20) for j in range(20,40)))
    toc = time.time()
    print("The parallel version took %f seconds" %(toc-tic))
    
    tic = time.time()
    for i in range(1,20):
        for j in range(20,40):
            _compute((i,j))
    toc = time.time()
    print("The serial version took %f seconds" %(toc-tic))


