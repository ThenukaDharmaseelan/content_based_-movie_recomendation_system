#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[26]:


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


# In[27]:


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# In[28]:


df = pd.read_csv('/Users/thenuka/Downloads/movie_recommender/movie_dataset.csv')


# In[29]:


df.head()


# In[30]:


print(df.columns)


# In[31]:


features = ['keywords','cast','genres','director']


# In[32]:


for feature in features:
    df[feature] = df[feature].fillna('')


# In[33]:


def combine_features(row):
    try:
        return row['keywords']+""+row['cast']+""+row["genres"]+""+row["director"]
    except:
        print ("Error:", row)


# In[34]:


df["combined_features"] = df.apply(combine_features, axis=1)


# In[35]:


print("combined_features"), df["combined_features"].head()


# In[36]:


cv = CountVectorizer()


# In[37]:


count_matrix = cv.fit_transform(df["combined_features"])


# In[38]:


count_matrix 


# In[40]:


cosine_sim = cosine_similarity(count_matrix)


# In[41]:


cosine_sim


# In[42]:


movie_user_likes = 'Aliens'


# In[43]:


movie_index = get_index_from_title(movie_user_likes)


# In[44]:


movie_index


# In[45]:


similar_movies = list(enumerate(cosine_sim[movie_index]))


# In[46]:


similar_movies


# In[47]:


sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse = True)


# In[48]:


sorted_similar_movies


# In[49]:


i = 0
for movie in sorted_similar_movies :
    print(get_title_from_index(movie[0]))
    i = i+1
    if i > 50:
        break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:





# In[ ]:





# In[23]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




