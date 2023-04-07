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
    
tfidf_vectorizer = TfidfVectorizer()

df_movies = pd.read_csv('movies.csv')
df_desc = pd.read_csv('movies_description.csv')

df_movies['year'] = df_movies.apply(getYear, axis=1)
movies = Cinemagoer()
K = 10

recommendations = []
selected = []

# getting top 10 movies for initial recommendation
# and show them
top250 = movies.get_top250_movies()
print('\nInitial recommendations:')
for i in range(K):
    recommendations.append(top250[i])
    print(top250[i]['title'])
    
while True:
    # show options
    print("\n1: Show recommendations\n2: Search movie\n3: Select movie\n4: View selected movies\n5: Exit program")
    option = int(input("Choose option: "))
    
    match option:
        case 1:
            # show recommendations
            print("\nMovie Recommendations:")
            for movie in recommendations:
                print(movie)
        case 2:
            # get search term
            term = str(input("\nSearch: "))
            searched = movies.search_movie(term)
            
            # show searched movies
            for movie in searched:
                print(movie.movieID, movie['title'])
        case 3:
            # get selection
            selection = str(input("\nSelection(movieID): "))
            selected_movie = movies.get_movie(selection)
            print("Selected ", selected_movie.movieID, selected_movie['title'])
            selected.append(selected_movie)
            
            # update recommendations
            genres_weighted_dictionary = {'total': 0}
            for movie in selected:
                for genre in movie['genres']:
                    if genre in genres_weighted_dictionary:
                        genres_weighted_dictionary[genre] += 1
                    else:
                        genres_weighted_dictionary[genre] = 1
                    genres_weighted_dictionary['total'] += 1
                    
            df_movies['multiple_metrics'] = df_movies.apply(lambda x: combined_metrics(genres_weighted_dictionary, selected, x), axis='columns')
            sorted_df = df_movies.sort_values(by='multiple_metrics', ascending=False)
            # drop the original movie selections from the results:
            for movie in selected:
                sorted_df.drop(sorted_df.loc[sorted_df['imdbId'] == movie.movieID[1:]].index, inplace=True)
            recommendations = sorted_df['title'].head(K).tolist()
        case 4:
            # show selected movies
            print("\nSelected movies:")
            for movie in selected:
                print(movie.movieID, movie['title'])
        case _:
            break

############################################################################
# END OF PROGRAM
############################################################################