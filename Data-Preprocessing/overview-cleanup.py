# extracts, cleans up overview data from movies_new.csv and writes new dataframe to movies_final.csv

import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords


# function for text cleaning
def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()

    return text


def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)

    # visualize words and frequencies
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=d, x="count", y="word")
    ax.set(ylabel='Word')
    plt.show()


# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


pd.set_option('display.max_colwidth', 300)

meta = pd.read_csv("movies_new.csv")

nltk.download('stopwords')

# creating data frame
movies = pd.DataFrame({'movie_name': meta['movie_name'], 'plot': meta['plot'], 'genre': meta['genre']})

movies['clean_plot'] = movies.apply(lambda x: clean_text(str(x['plot'])), axis=1)
# print(movies.head())

# freq_words(movies['clean_plot'], 100)

stop_words = set(stopwords.words('english'))

movies['clean_plot'] = movies['clean_plot'].apply(lambda x: remove_stopwords(x))
print(movies.head())

freq_words(movies['clean_plot'], 100)

movies.to_csv("movies_final.csv")


