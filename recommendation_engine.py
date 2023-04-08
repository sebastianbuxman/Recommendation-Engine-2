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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


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

recommendations = []
selected = []


def clustering(numClust, name):
    mlb = MultiLabelBinarizer()
    df_genre = pd.DataFrame(mlb.fit_transform(df_movies['genres']),
                            columns=mlb.classes_,
                            index=df_movies['genres'].index)

    # Scale the feature set
    scaled_features = StandardScaler().fit_transform(df_genre.values)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=numClust, random_state=0, n_init='auto')
    kmeans.fit(scaled_features)

    # Add the cluster labels to the dataset
    df_genre['cluster_labels'] = kmeans.labels_


    for i in range(numClust):
        '''if(name is in i):
            make recommendations from that cluster'''
    # Print the top 10 movies in each cluster
    for cluster_number in range(numClust):
        print(f"\n\nCluster Number {cluster_number}")
        print("====================")
        if()
        print(df_movies[df_genre['cluster_labels'] == cluster_number].head(10))


K = 10
# getting top 10 movies for initial recommendation
# and show them
top250 = movies.get_top250_movies()
print('\nInitial recommendations:')
for i in range(K):
    recommendations.append(top250[i])
    print(top250[i]['title'])

while True:
    option = int(input("\n1: Search Movie Titles \n2: Exit \n"))
    match option:
        case 1:
            term = str(input("Enter Movie title of choice:"))
            selected_movie = movies.search_movie(term)[0]

            # clustering

            # Below is giving the user an oppurtunity to choose how many clusters
            k = int(input("Choose value of k(must be greater than 2): "))
            clustering(k)
            # Weight to determine weight of each
            print("Choose the Weight distribution for each of the following")
            cos_weight = float(input("Enter the weight for cosine similarity (e.g. 0.8): "))
            lev_weight = float(input("Enter the weight for Levenshtein distance (e.g. 0.1): "))
            ed_weight = float(input("Enter the weight for Euclidean(e.g. 0.1): "))
            # weighted_sum = cos_weight * cos_result + lev_weight * lev_result + ed_weight * ed_result
            # Sort the recommendations by the weighted sum in descending order
            recommendations = sorted(recommendations, key=lambda x: x['weighted_sum'], reverse=True)
        case _:
            break
            # part 2
    base_case_desc = ""
    df_movies['multiple_metrics'] = df_movies.apply(lambda x: combined_metrics(base_case_desc, selected, x),
                                                    axis='columns')
    sorted_df = df_movies.sort_values(by='multiple_metrics', ascending=False)
    # drop the original movie selections from the results:
    for movie in selected:
        sorted_df.drop(sorted_df.loc[sorted_df['imdbId'] == movie.movieID[1:]].index, inplace=True)
    recommendations = sorted_df['title'].head(K).tolist()

############################################################################
# END OF PROGRAM
############################################################################
