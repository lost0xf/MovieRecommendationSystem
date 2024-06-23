#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


movies_data = pd.read_csv('movies.csv')


# In[3]:


movies_data.head()


# In[4]:


movies_data.tail()


# In[5]:


movies_data.shape


# In[6]:


selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']


# In[7]:


print(selected_features)


# In[8]:


for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# In[9]:


combined_features = movies_data['genres'] +' '+movies_data['keywords'] +' '+movies_data['tagline'] +' '+movies_data['cast'] +' '+movies_data['director'] 


# In[10]:


print(combined_features)


# In[11]:


vectorizer = TfidfVectorizer()


# In[12]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[13]:


print(feature_vectors)


# In[14]:


similarity = cosine_similarity(feature_vectors)


# In[15]:


print(similarity)


# In[16]:


print(similarity.shape)


# In[17]:


movie_name = input('Enter your favourite movie name :')


# In[18]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[19]:


find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[20]:


close_match = find_close_match[0]
print(close_match)


# In[21]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[22]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))


# In[23]:


print(similarity_score)


# In[24]:


len(similarity_score)


# In[25]:


sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_movies)


# In[26]:


print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if( i < 30):
        print(i, '.', title_from_index)
        i+=1


# In[28]:


movie_name = input('Enter your favourite movie name :')
list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if( i < 30):
        print(i, '.', title_from_index)
        i+=1


# In[ ]:




