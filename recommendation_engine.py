############################################################################
# Name: Max Beard
# Date: 03/26/2023
# Course: CS 3580
# Project: Recommendation Engine
# Description: a movie recommendation engine that is similar to Netflix
############################################################################

import pandas as pd
from imdb import Cinemagoer
import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity_function(base_case_desc, comparator_desc):
    # this line will convert the plots from strings to vectors in a single matrix:
    tfidf_matrix = tfidf_vectorizer.fit_transform((base_case_desc, comparator_desc))
    results = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return results[0][0]

def euclidean_distance(selections: list, comparator_year: int):
    result = 0
    for movie in selections:
        result += abs(int(movie['year']) - comparator_year)
    return result

def levenshtein_distance(selections: list, comparator_title: str):
    result = 0
    for movie in selections:
        result += Levenshtein.ratio(movie['title'], comparator_title)
    return result
    
def combined_metrics(base_case_desc: str, selections: list, comparator_movie: pd.core.series.Series):
    cs_result = cosine_similarity_function(base_case_desc, comparator_movie['overview'])
    ed_result = euclidean_distance(selections, int(comparator_movie['year']))
    lev_result = levenshtein_distance(selections, comparator_movie['title'])
    
    # normalize results
    cs_result = (cs_result + 1) / 2.0
    norm_ed_result = ed_result / 100
    
    return cs_result + lev_result - norm_ed_result

def getYear(row):
    year = row['title'][-5:-1]
    return year if year.isdigit() else 0

def getID(row):
    Id = 'tt0'+str(row)
    return Id
    
tfidf_vectorizer = TfidfVectorizer()

df_movies = pd.read_csv('movies.csv')
df_desc = pd.read_csv('movies_description.csv')
df_movies['imdb_id'] = df_movies['imdbId'].apply(getID)

df = df_movies.merge(df_desc, on='imdb_id')

df['year'] = df.apply(getYear, axis=1)
movies = Cinemagoer()
K = 10

recommendations = []
selected = []
    
while True:
    print("\n1: Search Movie Titles \n2: Exit")
    option = int(input("Choose option: "))
    match option:
        case 1:
            term = str(input("Enter Movie title of choice:"))
            selected_movie = movies.search_movie(term)[0]
            selected_movie_id = selected_movie.movieID[1:]
            
            # clustering
            
            #Below is giving the user an oppurtunity to choose how many clusters
            k = int(input("Choose value of k(must be greater than 2): "))
            #Weight to determine weight of each
            print("Choose the Weight distribution for each of the following")
            cos_weight = float(input("Enter the weight for cosine similarity (e.g. 0.8): "))
            lev_weight = float(input("Enter the weight for Levenshtein distance (e.g. 0.1): "))
            ed_weight = float(input("Enter the weight for Euclidean(e.g. 0.1): "))
            #weighted_sum = cos_weight * cos_result + lev_weight * lev_result + ed_weight * ed_result
            # Sort the recommendations by the weighted sum in descending order
            recommendations = sorted(recommendations, key=lambda x: x['weighted_sum'], reverse=True)
            
            # part 2
            base_case = df[(df['imdbId'] == int(selected_movie_id))]
            df['multiple_metrics'] = df.apply(lambda x: combined_metrics(base_case['overview'], selected, x), axis='columns')
            sorted_df = df.sort_values(by='multiple_metrics', ascending=False)
            # drop the original movie selections from the results:
            for movie in selected:
                sorted_df.drop(sorted_df.loc[sorted_df['imdbId'] == int(movie.movieID[1:])], inplace=True)
            recommendations = sorted_df['title'].head(K).tolist()
        case _:
            break 

############################################################################
# END OF PROGRAM
############################################################################