import pandas as pd
import numpy as np
# load numpy array from csv file
from numpy import loadtxt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import coo_matrix, hstack
# load array
# data = loadtxt('cosinesim.csv', delimiter=',')
books = pd.read_csv('books1.csv')
f = (pd.Series(books[['authors','title']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))
def getTitles():
    titles = books["title"].values
    return titles

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(f.astype('U'))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
titles = books['original_title']
indices = pd.Series(books.index, index=books['title'])
# def authors_recommendations(title):
#     idx = indices[title]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:21]
#     book_indices = [i[0] for i in sim_scores]
#     img_urls = [books['image_url'].iloc[i] for i in book_indices]
#     return [titles.iloc[book_indices],img_urls]

def authors_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:11]
    book_indices = [i[0] for i in sim_scores]
    rec = []
    for j in book_indices:
        dic = {}
        dic["title"] = books['title'].iloc[j]
        dic["img"] = books['image_url'].iloc[j]
        dic["year"] = books['original_publication_year'].iloc[j]
        dic["rating"] = books['average_rating'].iloc[j]
        dic["author"] = books['authors'].iloc[j].replace('  ','') 
        rec.append(dic)
    return rec[0],rec[1:]
