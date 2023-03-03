#%%
import pandas as pd 
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sklearn.feature_extraction.text import CountVectorizer

#creating a dataframe with movies name
movies=pd.read_csv('/Users/sumitsharma/Desktop/tmdb_5000_movies.csv',on_bad_lines='skip')
credits=pd.read_csv('/Users/sumitsharma/Desktop/tmdb_5000_credits.csv')
# print(movies)
movies.head()
credits.head(1)['cast'].values
#merging two dataframes
movies=movies.merge(credits,on='title')
print(movies.shape)
#important columns in the merged dataframe are genres, id,keywords,title,overview,cast,crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']].sort_values(by='movie_id')

movies.dropna(inplace=True)
movies.isnull().sum()
#checking if there is duplicate data present
# movies.duplicated().sum()
#data preprocessing 
movies.iloc[0].genres
#Helper function to convert genres text into a set of keywords
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj): #ast.literaleval converts string into list
        l.append(i['name'])
        return l
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)

#helper function to get names of first three actors in cast
def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
          L.append(i['name'])
          counter+=1
        else:
            break
    return L
movies['cast']=movies['cast'].apply(convert3)
def fetch_director(obj):
    lis=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            lis.append(i['name'])
            break
    return lis
movies['crew']=movies['crew'].apply(fetch_director)
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
#function to convert movies overview from a string to list 
movies['overview']=movies['overview'].apply(collapse)
movies['cast']=movies['cast'].apply(collapse)
movies['crew']=movies['crew'].apply(collapse)
movies['genres']=movies['genres'].apply(collapse)
movies['keywords']=movies['keywords'].apply(collapse)
print(movies.head)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()
#converting the concatenated list to string
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()
cv = CountVectorizer(max_features=5000,stop_words='english')
    
vector = cv.fit_transform(new['tags']).toarray()
vector.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

new[new['title'] == 'The Lego Movie'].index[0]
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        
    
recommend('Gandhi')