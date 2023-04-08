import pandas as pd
from imdb import Cinemagoer
import Levenshtein
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import tkinter as tk
from tkinter import messagebox

# Functions -----------------------------------------------------------------------------------------
def cosine_similarity_function(base_case_desc: str, comparator_desc: str):
    # this line will convert the descriptions from strings to vectors in a single matrix:
    tfidf_matrix = tfidf_vectorizer.fit_transform((base_case_desc, comparator_desc))
    results = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return results[0][0]

def euclidean_distance(base_case_year: int, comparator_year: int):
    return abs(base_case_year - comparator_year)

def levenshtein_distance(base_case_title: str, comparator_title: str):
    return Levenshtein.ratio(base_case_title, comparator_title)
    
def combined_metrics(base_case: pd.core.series.Series, comparator_movie: pd.core.series.Series):
    cs_result = cosine_similarity_function(base_case['overview'], comparator_movie['overview'])
    ed_result = euclidean_distance(int(base_case['year']), int(comparator_movie['year']))
    lev_result = levenshtein_distance(base_case['title'], comparator_movie['title'])
    
    # normalize results
    cs_result = (cs_result + 1) / 2.0
    norm_ed_result = ed_result / 100
    
    # getting weights
    cs_weight = float(cosine_weight_entry.get()) if cosine_var.get() else 0
    lev_weight = float(levenshtein_weight_entry.get()) if levenshtein_var.get() else 0
    ed_weight = float(euclidean_weight_entry.get()) if euclidean_var.get() else 0
    
    return cs_result*cs_weight + lev_result*lev_weight - norm_ed_result*ed_weight

def getYear(row):
    year = row['title'][-5:-1]
    return year if year.isdigit() else 0

def getID(row):
    stringRow = str(row)
    while len(stringRow) < 7:
        stringRow = "0" + stringRow
    return "tt" + stringRow

def clustering(numClust, name):
    # convert the string of genres to a list:
    df['genres_list'] = [x.split('|') for x in df['genres']]
    
    mlb = MultiLabelBinarizer()
    df_genre = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                            columns=mlb.classes_,
                            index=df['genres'].index)

    # Scale the feature set
    scaled_features = StandardScaler().fit_transform(df_genre.values)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=numClust, random_state=0, n_init='auto')
    kmeans.fit(scaled_features)
    
    df_genre = pd.concat([df[['original_title', 'imdbId']], df_genre], axis=1)

    # Add the cluster labels to the dataset
    df_genre['cluster_labels'] = kmeans.labels_

    if name not in df['original_title'].values:
        print(f"\nThe movie '{name}' is not in the dataframe.")
        return

    # Check if the movie name entered by the user is in a specific cluster
    cluster_number = df_genre.loc[df['title'] == name, 'cluster_labels'].iloc[0]
    print(f"\nThe movie '{name}' is in cluster number {cluster_number}")
    # Make recommendations from the specific cluster
    cluster_movies = df[df_genre['cluster_labels'] == cluster_number]
    print(f"\nRecommendations from cluster number {cluster_number}")
    print("====================")
    print(cluster_movies.head(10))

    '''for i in range(numClust):
        if(name is in i):
            make recommendations from that cluster
    # Print the top 10 movies in each cluster
    for cluster_number in range(numClust):
        print(f"\n\nCluster Number {cluster_number}")
        print("====================")
        print(df[df_genre['cluster_labels'] == cluster_number].head(10))'''

def get_recommendations():
    # get the input values from the text boxes and checkboxes
    movie = movies.search_movie(movie_entry.get())[0]
    movie_id = movie.movieID[1:]
    K = int(cluster_entry.get())
    clustering(K, movie)

    # perform the recommendation logic here using the input values
    base_case = df[(df['imdbId'] == int(movie_id))].iloc[0]
    df['multiple_metrics'] = df.apply(lambda x: combined_metrics(base_case, x), axis='columns')
    sorted_df = df.sort_values(by='multiple_metrics', ascending=False)
    # drop first recommendation since it is the selected movie
    sorted_df.drop([0], inplace=True)
    # display the recommendations to the user
    recommendations = sorted_df['original_title'].head().tolist()
    messagebox.showinfo("Recommendations", '\n'.join(recommendations))

# Data processing ---------------------------------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer()

df_movies = pd.read_csv('movies.csv')
df_desc = pd.read_csv('movies_description.csv')
df_movies['imdb_id'] = df_movies['imdbId'].apply(getID)

df = df_movies.merge(df_desc, on='imdb_id')

df['year'] = df.apply(getYear, axis=1)
df.fillna("", inplace=True)
movies = Cinemagoer()

# GUI application ---------------------------------------------------------------------------------
root = tk.Tk()
root.title("Movie Recommendation System")
mainframe = tk.Frame(root, padx=20, pady=20)
mainframe.grid(column=0, row=0)

# create labels and text boxes for movie title and number of clusters
movie_label = tk.Label(mainframe, text="Movie title:")
movie_label.grid(row=0, column=0)
movie_entry = tk.Entry(mainframe)
movie_entry.grid(row=0, column=1)

cluster_label = tk.Label(mainframe, text="Number of clusters:")
cluster_label.grid(row=1, column=0)
cluster_entry = tk.Entry(mainframe)
cluster_entry.grid(row=1, column=1)

# create checkboxes and entry boxes for distance metrics and weights
cosine_var = tk.IntVar()
cosine_check = tk.Checkbutton(mainframe, text="Cosine similarity", variable=cosine_var)
cosine_check.grid(row=2, column=0)

cosine_weight_label = tk.Label(mainframe, text="Weight:")
cosine_weight_label.grid(row=2, column=1)
cosine_weight_entry = tk.Entry(mainframe)
cosine_weight_entry.grid(row=2, column=2)

levenshtein_var = tk.IntVar()
levenshtein_check = tk.Checkbutton(mainframe, text="Levenshtein distance", variable=levenshtein_var)
levenshtein_check.grid(row=3, column=0)

levenshtein_weight_label = tk.Label(mainframe, text="Weight:")
levenshtein_weight_label.grid(row=3, column=1)
levenshtein_weight_entry = tk.Entry(mainframe)
levenshtein_weight_entry.grid(row=3, column=2)

euclidean_var = tk.IntVar()
euclidean_check = tk.Checkbutton(mainframe, text="Euclidean distance", variable=euclidean_var)
euclidean_check.grid(row=4, column=0)

euclidean_weight_label = tk.Label(mainframe, text="Weight:")
euclidean_weight_label.grid(row=4, column=1)
euclidean_weight_entry = tk.Entry(mainframe)
euclidean_weight_entry.grid(row=4, column=2)

# create button for getting recommendations
recommend_button = tk.Button(mainframe, text="Get Recommendations", command=get_recommendations)
recommend_button.grid(row=5, column=1)

root.mainloop()
