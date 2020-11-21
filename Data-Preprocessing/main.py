import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_colwidth', 300)

meta = pd.read_csv("movies_metadata.csv.csv")

# creating data frame
movies = pd.DataFrame({'movie_name': meta['movie_name'], 'plot': meta['plot']})

# an empty list
genres = []
countGenres = []

# extract genres
for i in meta['genre']:
    genre = []
    li = i[1:-1].split(", ")
    for j in li:
        if j.startswith("'name': "):
            j = j[9:-2]
            if j != 'Drama' and j != 'Comedy':
                genre.append(j)
    countGenres.append(len(genre))
    genres.append(genre)

# add to 'movies' dataframe
movies['genre'] = genres

# to check how many movies have no genres
movies_new = movies[~(movies['genre'].str.len() == 0)]
print(movies_new.shape, movies.shape)

# to get frequency distribution of genre labels
# get all genre tags in a list
all_genres = sum(genres, [])
print(set(all_genres))

all_genres = nltk.FreqDist(all_genres)

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),
                              'Count': list(all_genres.values())})
g = all_genres_df.nlargest(columns="Count", n=50)
plt.figure(figsize=(12, 15))
ax = sns.barplot(data=g, x="Count", y="Genre")
ax.set(ylabel='Count')
plt.show()

# to get frequency distribution of number of genres per movie
countGenres = nltk.FreqDist(countGenres)

# create dataframe
genre_count_df = pd.DataFrame({'Movie': list(countGenres.keys()),
                               'Count': list(countGenres.values())})
g = genre_count_df.nlargest(columns="Count", n=50)
plt.figure(figsize=(12, 15))
ax = sns.barplot(data=g, x="Movie", y="Count")
ax.set(ylabel='Count')
plt.show()

movies.to_csv("movies_new.csv")

