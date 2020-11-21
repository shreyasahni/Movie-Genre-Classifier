import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st


def print_distr(genreList):
    genres = []

    for i in genreList:
        li = i[1:-1].split(", ")
        genres.append(li)

    all_genres = sum(genres, [])

    all_genres = nltk.FreqDist(all_genres)

    # create dataframe
    all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),
                                  'Count': list(all_genres.values())})
    g = all_genres_df.nlargest(columns="Count", n=50)
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=g, x="Count", y="Genre")
    ax.set(ylabel='Count')
    plt.show()


pd.set_option('display.max_colwidth', 300)

meta = pd.read_csv("movies_final.csv")

# creating data frame
movies = pd.DataFrame({'movie_name': meta['movie_name'], 'plot': meta['clean_plot'], 'genre': meta['genre']})

# getting some statistics
sizes = []

for i in movies['plot']:
    sizes.append(len(str(i).split()))

print("Mean: ", st.mean(sizes))
print("Median: ", st.median(sizes))
print("Mode: ", st.mode(sizes))

sizes = nltk.FreqDist(sizes)

# create dataframe
all_sizes_df = pd.DataFrame({'Size': list(sizes.keys()),
                             'Count': list(sizes.values())})
g = all_sizes_df.nlargest(columns="Count", n=100)
plt.figure(figsize=(30, 15))
ax = sns.barplot(data=g, x="Size", y="Count")
ax.set(ylabel='Count')
plt.show()

# randomising data set
movies = movies.sample(frac=1)

# splitting data set
df_1 = movies.iloc[:25000, :]
df_2 = movies.iloc[25001:, :]

df_1.to_csv("movies_train.csv")
df_2.to_csv("movies_test.csv")

print_distr(df_1['genre'])
print_distr(df_2['genre'])

