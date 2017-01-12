#Author: stefano1.cereda@mail.polimi.it

import pandas as pd
import numpy as np

FILENAME_HETREC_BASE = "../datasets/hetrec/"
FILENAME_MOVIESDAT = FILENAME_HETREC_BASE + "movies.dat"
FILENAME_MOVIE_GENRES = FILENAME_HETREC_BASE + "movie_genres.dat"
FILENAME_MOVIE_DIRECTORS = FILENAME_HETREC_BASE+ "movie_directors.dat"
FILENAME_MOVIE_ACTORS = FILENAME_HETREC_BASE+ "movie_actors.dat"
FILENAME_MOVIE_COUNTRIES = FILENAME_HETREC_BASE+ "movie_countries.dat"
FILENAME_MOVIE_LOCATIONS = FILENAME_HETREC_BASE+ "movie_locations.dat"

def _read_movies():
    #someone must hate utf...
    movies = pd.read_csv(FILENAME_MOVIESDAT, sep='\t', header=0)
    movies = movies.drop(('imdbID', 'spanishTitle', 'imdbPictureURL', 'rtID', 
                          'rtAllCriticsRating', 'rtAllCriticsNumReviews', 'rtAllCriticsNumFresh',
                          'rtAllCriticsNumRotten', 'rtAllCriticsScore', 'rtTopCriticsRating',
                          'rtTopCriticsNumReviews', 'rtTopCriticsNumFresh', 'rtTopCriticsNumRotten',
                          'rtAudienceRating', 'rtAudienceNumRatings', 'rtAudienceScore', 'rtPictureURL'),1)
    return movies


def _read_movies_genres():
    genres = pd.read_csv(FILENAME_MOVIE_GENRES, sep='\t', header=0)

    available_genres = genres['genre'].unique()
    genres_mask = np.ones(available_genres.shape[0], dtype=bool)

    #which is the fast way to do this?
    genres_dict = {}
    for movie in genres['movieID'].unique():
        genres_dict[movie] = genres_mask.copy()
        tagged_genres = genres[genres['movieID'] == movie]['genre'].values
        
        for i,genre in enumerate(available_genres):
            genres_dict[movie][i] = genre in tagged_genres

    return genres_dict


def _read_movies_directors():
    directors = pd.read_csv(FILENAME_MOVIE_DIRECTORS, sep='\t', header=0)
    directors = directors.drop('directorName')
    return directors


###WARNING: WE HAVE INFORMATIONS JUST FOR 29 MOVIES!!!
def _read_movie_actors():
    actors = pd.read_csv(FILENAME_MOVIE_ACTORS, sep='\t', header=0)

    available_actors = actors['actorID'].unique()
    actors_vect = np.zeros(available_actors.shape[0], dtype=int)

    #which is the fast way to do this?
    actors_dict = {}
    for movie in actors['movieID'].unique():
        actors_dict[movie] = actors_vect.copy()
        featured_actors = actors[actors['movieID'] == movie][['actorID','ranking']]
        
        for i,actor in enumerate(available_actors):
            if actor in featured_actors['actorID'].values:
                rank = featured_actors[featured_actors['actorID'] == actor]['ranking'].values[0]
                if np.isnan(rank):
                    rank = 0
                actors_dict[movie][i] = int(rank)

    return actors_dict


def _read_movie_countries():
    countries = pd.read_csv(FILENAME_MOVIE_COUNTRIES, sep='\t', header=0)
    return countries


#TODO
#WARNING: I did not clearly understand how the data are represented
def _read_movie_locations():
    locations = pd.read_csv(FILENAME_MOVIE_LOCATIONS, sep='\t', header=0)
    locations[locations['location4'] == 'NaN']#?
    available_locations = np.unique(locations[['location1','location2','location3','location4']].values.ravel())
    return None
