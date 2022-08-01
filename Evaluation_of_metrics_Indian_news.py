#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install sentence_transformers


# In[52]:


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')


# In[53]:


import pandas as pd
df = pd.read_csv("hindu_news_climate.csv")


# In[54]:


df = df.dropna()


# In[55]:


import regex as re
import nltk
from nltk import word_tokenize
def clean_text(text):
    text = text.lower()                                  # lower-case all characters
    text =  re.sub(r'pic.\S+', '',text) 
    text =  re.sub(r"[^a-zA-Z+']", ' ',text)             # only keeps characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ')      # keep words with length>1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.tokenize.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english') + ["india","said","climate","change","hindu","news"]   # remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i)>2])
    text= re.sub("\s[\s]+", " ",text).strip()            # remove repeated/leading/trailing spaces
    return text


# In[56]:


import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


# In[57]:


import string
df.text = df.text.apply(clean_text).apply(lemmatize_text)
df.head()
df.shape


# In[58]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[62]:


bert_encodings = model.encode(df.text.values)


# In[63]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
# Fit K-Means
kmeans_1 = KMeans(n_clusters=8,random_state= 10)
# Use fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(bert_encodings)
# Calculate cluster validation metrics
score_kemans_s = silhouette_score(bert_encodings, kmeans_1.labels_, metric='euclidean')
score_kemans_c = calinski_harabasz_score(bert_encodings, kmeans_1.labels_)
score_kemans_d = davies_bouldin_score(bert_encodings, predictions)
print('Silhouette Score: %.4f' % score_kemans_s)
print('Calinski Harabasz Score: %.4f' % score_kemans_c)
print('Davies Bouldin Score: %.4f' % score_kemans_d)
print("The above scores are for Indian news using Kmeans and optimum cluster")


# In[64]:


# Inter cluster distance map
from yellowbrick.cluster import InterclusterDistance
# Instantiate the clustering model and visualizer
visualizer = InterclusterDistance(kmeans_1)
visualizer.fit(df)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[65]:


# gaussian mixture clustering
from numpy import unique
from numpy import where
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
# define the model
model = GaussianMixture(n_components= 9,covariance_type= "full", random_state = 10)
# fit the model
model.fit(bert_encodings)
# assign a cluster to each example
yhat = model.predict(bert_encodings)
# retrieve unique clusters
clusters = unique(yhat)
# Calculate cluster validation score
score_dbsacn_s = silhouette_score(bert_encodings, yhat, metric='euclidean')
score_dbsacn_c = calinski_harabasz_score(bert_encodings, yhat)
score_dbsacn_d = davies_bouldin_score(bert_encodings, yhat)
print('Silhouette Score: %.4f' % score_dbsacn_s)
print('Calinski Harabasz Score: %.4f' % score_dbsacn_c)
print('Davies Bouldin Score: %.4f' % score_dbsacn_d)


# In[66]:


# Agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
# define the model
model = AgglomerativeClustering(n_clusters=9)
# fit model and predict clusters
yhat = model.fit(bert_encodings)
yhat_2 = model.fit_predict(bert_encodings)
# retrieve unique clusters
clusters = unique(yhat)
# Calculate cluster validation metrics
score_AGclustering_s = silhouette_score(bert_encodings, yhat.labels_, metric='euclidean')
score_AGclustering_c = calinski_harabasz_score(bert_encodings, yhat.labels_)
score_AGclustering_d = davies_bouldin_score(bert_encodings, yhat_2)
print('Silhouette Score: %.4f' % score_AGclustering_s)
print('Calinski Harabasz Score: %.4f' % score_AGclustering_c)
print('Davies Bouldin Score: %.4f' % score_AGclustering_d)


# DBSCAN **EVALUATION**

# In[67]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
neighbors = NearestNeighbors(n_neighbors=550)
neighbors_fit = neighbors.fit(bert_encodings)
distances, indices = neighbors_fit.kneighbors(bert_encodings)


# In[68]:


import numpy as np
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for Indian news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[69]:


# dbscan clustering
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# define dataset
# define the model
model = DBSCAN(eps=0.42, min_samples= 3)
# rule of thumb for min_samples: 2*len(cluster_df.columns)
# fit model and predict clusters
yhat = model.fit_predict(bert_encodings)
# retrieve unique clusters
clusters = unique(yhat)
# Calculate cluster validation metrics
score_dbsacn_s = silhouette_score(bert_encodings, yhat, metric='euclidean')
score_dbsacn_c = calinski_harabasz_score(bert_encodings, yhat)
score_dbsacn_d = davies_bouldin_score(bert_encodings, yhat)
print('Silhouette Score: %.4f' % score_dbsacn_s)
print('Calinski Harabasz Score: %.4f' % score_dbsacn_c)
print('Davies Bouldin Score: %.4f' % score_dbsacn_d)


# In[ ]:





# In[ ]:




