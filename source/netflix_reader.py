import scipy.io as sio
import numpy as np
from difflib import SequenceMatcher

BASEFILE = "../datasets/Enriched_Netflix_Dataset/"

icm = sio.loadmat(BASEFILE + "./icm.mat")
icm_matrix = icm['icm']
icm_dictionary = icm['dictionary']
icm_stems = icm_dictionary['stems'][0][0]
icm_stemtypes = icm_dictionary['stemTypes'][0][0]

titles = sio.loadmat(BASEFILE + "./titles.mat")['titles']

urm = sio.loadmat(BASEFILE + "./urm.mat")['urm']


#Now let's try to aggregate the features

#The first group is composed by the actors
actor_features_names = ['ActorsLastNameFirstArray', 'ActorsLastNameFirstArray;DirectorsLastNameFirstArray']
actor_features = list()
for feature_name in actor_features_names:
    actor_features.extend(np.where(icm_stemtypes == feature_name)[0].tolist())
#you can now obtain the relevant icm with print(icm_matrix[actor_features]) and obtain the actors featured in a movie with print(icm_matrix[actor_features, movieId])


#the second group of features is the country of production
county_features_names = ['ChannelName;CountryOfOrigin', 'CountryOfOrigin']
country_features = list()
for feature_name in county_features_names:
    country_features.extend(np.where(icm_stemtypes == feature_name)[0].tolist())
    
    
#the third group of features is the director, with just one name 
director_features = np.where(icm_stemtypes == 'DirectorsLastNameFirstArray')[0].tolist()


#we now have the genres.
#TODO: Bear in mind that some of these features are extracted from tags/stems so it could be worth separating them
genres_features_names = ['GenresArray', 'GenresArray;KeywordsArray', 'GenresArray;KeywordsArray;TitleFull', 'GenresArray;TitleFull']
genres_features = list()
for feature_name in genres_features_names:
    genres_features.extend(np.where(icm_stemtypes == feature_name)[0].tolist())


#skip the tags (for now) TODO and go to miniseries
miniseries_features = np.where(icm_stemtypes == 'ShowType')[0].tolist()


#Skip the title as we have a separate file fo them and do the year
years_features = np.where(icm_stemtypes == 'Year')[0].tolist()





#compute f_{i;actors} * f_{j;actors} by counting the number of common actors
#TODO: is better to use the number of common actors or normalize it againsta the number of featured actors?
def similarity_actors(movieId1, movieId2):
    actors_1 = icm_matrix[actor_features, movieId1].astype(bool)
    actors_2 = icm_matrix[actor_features, movieId2].astype(bool)

    num_act = np.sum(actors_1) + np.sum(actors_2)
    num_commons = np.sum(actors_1.multiply(actors_2))
    
    ret num_commons


#compute the similarity of the origin country by checking if they are the same
def similarity_countries(movieId1, movieId2):
    country_1 = icm_matrix[country_features, movieId1].astype(bool)
    country_2 = icm_matrix[country_features, movieId2].astype(bool)

    return np.sum(country_1.multiply(country_2))


#check if the director is the same
def similarity_directors(movieId1, movieId2):
    dir_1 = icm_matrix[director_features, movieId1].astype(bool)
    dir_2 = icm_matrix[director_features, movieId2].astype(bool)

    return np.sum(dir_1.multiply(dir_2))


#The similarity between genres is given by the number of common genres
#TODO: like for the actors, would a percentage be better?
def similarity_genres(movieId1, movieId2):
    genres_1 = icm_matrix[genres_features, movieId1].astype(bool)
    genres_2 = icm_matrix[genres_features, movieId2].astype(bool)

    num_genres = np.sum(genres_1) + np.sum(genres_2)
    num_commons = np.sum(genres_1.multiply(genres_2))
    
    ret num_commons
    
    
#check if the two items are both miniseries OR if they are both NOT miniseries
def similarity_miniseries(movieId1, movieId2):
    type_1 = icm_matrix[miniseries_features, movieId1].astype(bool).toarray()[0][0]
    type_2 = icm_matrix[miniseries_features, movieId2].astype(bool).toarray()[0][0]

    return np.sum(type_1 == type_2)


#The similarity of two years is the inverse of their difference
#TODO: maybe try a more refined approach
def similarity_years(movieId1, movieId2):
    year_feat_1 = icm_matrix[years_features, movieId1].astype(bool).indices[0]
    year_feat_2 = icm_matrix[years_features, movieId2].astype(bool).indices[0]
    
    year_1 = int(icm_stems[years_features[year_feat_1]][0][0])
    year_2 = int(icm_stems[years_features[year_feat_2]][0][0])

    return 1/(abs(year_1-year_2))
    

#The similarity of two titles is the number of common words
def similarity_titles(movieId1, movieId2):
    title_1 = titles[movieId1]
    title_2 = titles[movieId2]
    return SequenceMatcher(lambda x: x in " \t", title_1, title_2).ratio() #skip the blanks
