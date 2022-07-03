import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,jsonify
import re, string, random


app = Flask(__name__)

    
df = pd.read_csv("netflix_titles.csv")
df.head()
df.shape
a = np.unique(df['country'].apply(str))# .isnull().sum()
# print(f'Total countries:{a}')
df_1 = df.loc[(df['country'] == 'India') & (df['release_year']>2000)]
# print(len(df_1))
my_df = pd.DataFrame(df_1)
finaldata = my_df[["title","description"]]
finaldata = finaldata.set_index('title')

# NLP

lemmatizer = WordNetLemmatizer()
def preprocess(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    sentence = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    finalsent = ' '.join(sentence)
    
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    
    return finalsent

finaldata["new_description"]= finaldata["description"].apply(preprocess)
c = finaldata.head()
# print(f'new dec:{c}')

finaldata = my_df[["title","description"]]
finaldata = finaldata.set_index('title')

a = np.unique(df['title'])
# print(a)
finaldata.tail()

# Sklearn

tfidf = TfidfVectorizer()
tfidf_movieid = tfidf.fit_transform((finaldata["description"]))

similarity = cosine_similarity(tfidf_movieid, tfidf_movieid)
indices = pd.Series(finaldata.index)

def recommendations(title, cosine_sim = similarity):
    try:
        index = indices[indices == title].index[0]
        print(index)
        similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending = False)
        print(similarity_scores)
        top_10_movies = list(similarity_scores.iloc[1:20].index)
        print(top_10_movies)
        recommended_movies = [list(finaldata.index)[i] for i in top_10_movies]
        return recommended_movies
    except:
        print("No movie name found")  

                                    

