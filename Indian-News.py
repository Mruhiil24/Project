#!/usr/bin/env python
# coding: utf-8

# ## Data Cleaning

# In[1]:


import string
import pandas as pd 
import numpy as np
df = pd.read_csv("hindu_news_climate.csv")
df1 = pd.read_csv("nbc_news.csv")
df2 = pd.read_csv("ny_times.csv")
df3 = pd.read_csv("bbc_news_climate.csv")
f = pd.concat([df1, df2, df3])


# In[2]:


f


# In[3]:


df


# In[4]:


df2


# In[5]:


df3


# In[6]:


df.head()


# In[7]:


df = df.dropna()
f = f.dropna()


# In[8]:


df = df[df["text"]!=""]


# In[9]:


df.shape


# In[10]:


f.shape


# In[11]:


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


# In[12]:


import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


# In[13]:


df.text = df.text.apply(clean_text).apply(lemmatize_text)
df.head()


# In[14]:


f.text = f.text.apply(clean_text).apply(lemmatize_text)
f.head()


# # Vectorization

# ## Bag of Words

# In[15]:


from sklearn.feature_extraction.text import  CountVectorizer
count = CountVectorizer(min_df=3, max_features=1000)
count1 = CountVectorizer(min_df=10, max_features=1000)
bow_vectors = count.fit_transform(df["text"].values)
bow1 = count1.fit_transform(f["text"].values)
#since there is no train and test, we are using fit_transform on all the data


# In[16]:


bow_vectors.shape


# In[17]:


bow1.shape


# ## TFIDF

# In[18]:


from sklearn.feature_extraction.text import  TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=3, max_features=1000)
vectorizer1 = TfidfVectorizer(min_df=10, max_features=1000)
tf_idf_vectors = vectorizer.fit_transform(df["text"].values)
tf_idf1 = vectorizer1.fit_transform(f["text"].values)


# In[19]:


tf_idf_vectors.shape


# In[20]:


bow1.shape


# ## Word Vectors

# In[21]:


# pip install wget


# In[22]:


get_ipython().system('wget https://s3.amazonaws.comhttps://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')
# !wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz


# In[23]:


import gzip
import shutil


# In[24]:


with gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb') as f_in:
    with open('GoogleNews-vectors-negative300.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[25]:


# pip install --upgrade gensim


# In[26]:


import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)


# In[27]:


def compute_WordVectors(sents,model):
    
    vectors = []
    for document in sents:        
        sentence_vector = 0
        count =0
        for word in document.split():
            try:
                sentence_vector = sentence_vector + model[word]
                count = count+1
            except KeyError:
                continue
        
        sentence_vector = sentence_vector/count
        vectors.append(sentence_vector)
    
    return np.vstack(vectors)


# In[28]:


word_vectors = compute_WordVectors(df["text"].values,model)


# In[29]:


word_vectors1 = compute_WordVectors(f["text"].values,model)


# In[30]:


word_vectors.shape


# In[31]:


word_vectors1.shape


# # Clustering

# ## Kmeans

# ## Kmeans with Bag of Words

# In[32]:


from sklearn.cluster import KMeans
n = list(range(2,40,2))
inertia = []
inertia1 = []
for i in n:
    
    clusterer = KMeans(i)
    clusterer1 = KMeans(i)
    
    clusterer.fit(bow_vectors)
    clusterer1.fit(bow1)
    inertia.append(clusterer.inertia_)
    inertia1.append(clusterer1.inertia_)


# In[33]:


import matplotlib.pyplot as plt

plt.plot(n,inertia)
plt.title("KMeans with Bag of words for Indian news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[34]:


plt.plot(n,inertia1)
plt.title("KMeans with Bag of words for american news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[35]:


clusterer = KMeans(8)
    
cluster_labels = clusterer.fit_predict(bow_vectors)


# In[36]:


clusterer1 = KMeans(8)

cluster1_labels = clusterer1.fit_predict(bow1)


# In[37]:


df["labels"] = cluster_labels
# f["labels"] = cluster1_labels 
idx_to_word = {values:key for key,values in count.vocabulary_.items()}

vocab_idx = np.argsort(clusterer.cluster_centers_, axis=1)[:,::-1][:,:10]


for i in range(8):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# In[38]:


cluster1_labels.shape


# In[39]:


# df["labels"] = cluster_labels
f["labels"] = cluster1_labels 
idx_to_word = {values:key for key,values in count1.vocabulary_.items()}

vocab_idx = np.argsort(clusterer1.cluster_centers_, axis=1)[:,::-1][:,:10]


for i in range(8):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# ## Kmeans with TFIDF Vectors

# In[ ]:





# In[40]:


from sklearn.cluster import KMeans
n = list(range(2,40,2))
inertia = []
inertia1 = []
for i in n:
    
    clusterer = KMeans(i)
    clusterer1 = KMeans(i)
    clusterer.fit(tf_idf_vectors)
    clusterer1.fit(tf_idf1)
    inertia.append(clusterer.inertia_)
    inertia1.append(clusterer1.inertia_)


# In[41]:


import matplotlib.pyplot as plt

plt.plot(n,inertia)
plt.title("KMeans with TFIDF")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[42]:


import matplotlib.pyplot as plt

plt.plot(n,inertia1)
plt.title("KMeans with TFIDF for american news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[43]:


clusterer = KMeans(9)
    
cluster_labels = clusterer.fit_predict(tf_idf_vectors)

clusterer1 = KMeans(6)

cluster1_labels = clusterer1.fit_predict(tf_idf1)


# In[44]:


df["labels"] = cluster_labels
f["labels"] = cluster1_labels


# In[45]:


df.head()


# In[46]:


f.head()


# In[47]:


idx_to_word = {values:key for key,values in vectorizer.vocabulary_.items()}


# In[48]:


vocab_idx = np.argsort(clusterer.cluster_centers_, axis=1)[:,::-1][:,:10]


# In[49]:


for i in range(9):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# In[50]:


idx_to_word = {values:key for key,values in vectorizer1.vocabulary_.items()}
vocab_idx = np.argsort(clusterer1.cluster_centers_, axis=1)[:,::-1][:,:10]


# In[51]:


for i in range(6):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# ## Kmeans with Word Vectors

# In[ ]:





# In[52]:


from sklearn.cluster import KMeans
n = list(range(2,40,2))
inertia = []
inertia1 = []
for i in n:
    
    clusterer = KMeans(i)
    clusterer1 = KMeans(i)
    clusterer.fit(word_vectors)
    clusterer1.fit(word_vectors1)
    inertia.append(clusterer.inertia_)
    inertia1.append(clusterer1.inertia_)

plt.plot(n,inertia)
plt.title("KMeans with Word Vectors for Indian news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[53]:


plt.plot(n,inertia1)
plt.title("KMeans with Word Vectors for american news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[54]:


clusterer = KMeans(4)
    
cluster_labels = clusterer.fit_predict(word_vectors)

clusterer1 = KMeans(4)

cluster1_labels = clusterer1.fit_predict(word_vectors1)


# In[55]:


df["labels"] = cluster_labels
f["labels"] = cluster1_labels


# In[56]:


df.head()


# In[57]:


f.head()


# In[58]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update(["country", "year", "state", "world", "time","one","area","people","que","los","la","para","una",
"por","del","con","dijo","como"])


# In[59]:


for i in range(4):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[60]:


for i in range(4):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(f[f["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[ ]:





# ## GaussianMixture

# ## GaussianMixture with Bag of Words

# In[61]:


from sklearn.mixture import GaussianMixture
from sklearn import mixture
# We use a small scope of grid search to find the best parameters of GMM

lowest_bic = np.infty   # initial BIC is set to infinity
best_gmm = None
lowest_bic1 = np.infty   # initial BIC is set to infinity
best_gmm1 = None
n_components_range = range(2,10)    # The number of clusters

cv_types = ['spherical', 'tied', 'diag']  # The covariance type

for cv_type in cv_types:
    
    for n_components in n_components_range:
        
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        gmm1 = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        gmm.fit(bow_vectors.toarray())
        gmm1.fit(bow1.toarray())
        bic = gmm.bic(bow_vectors.toarray())  # get Model BIC
        bic1 = gmm1.bic(bow1.toarray())
        if bic < lowest_bic:  # save the model with lowest BIC sofar
            lowest_bic = bic
            best_gmm = gmm
        if bic1 < lowest_bic1:
            lowest_bic1 = bic1
            best_gmm1 = gmm1


# In[62]:


lowest_bic
best_gmm


# In[63]:


lowest_bic1
best_gmm1


# In[65]:


clusterer = GaussianMixture(9)
    
cluster_labels = clusterer.fit_predict(bow_vectors.toarray())

clusterer1 = GaussianMixture(7)

cluster1_labels = clusterer1.fit_predict(bow1.toarray())


# In[66]:


clusterer.weights_.argsort()[::-1][:5]


# In[67]:


clusterer1.weights_.argsort()[::-1][:5]


# In[68]:


df["labels"] = cluster_labels

idx_to_word = {values:key for key,values in count.vocabulary_.items()}

vocab_idx = np.argsort(clusterer.means_, axis=1)[:,::-1][:,:10]


#picking  5 important clusters based on weights
for i in clusterer.weights_.argsort()[::-1][:5]:
    print("="*50)
    print("Cluster ",i," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# In[69]:


f["labels"] = cluster1_labels

idx_to_word = {values:key for key,values in count1.vocabulary_.items()}

vocab_idx = np.argsort(clusterer1.means_, axis=1)[:,::-1][:,:10]


#picking  5 important clusters based on weights
for i in clusterer1.weights_.argsort()[::-1][:5]:
    print("="*50)
    print("Cluster ",i," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# ### GaussianMixture with TFIDF

# In[70]:


from sklearn.mixture import GaussianMixture
from sklearn import mixture
# We use a small scope of grid search to find the best parameters of GMM

lowest_bic = np.infty   # initial BIC is set to infinity
best_gmm = None
lowest_bic1 = np.infty   # initial BIC is set to infinity
best_gmm1 = None
n_components_range = range(2,10)    # The number of clusters

cv_types = ['spherical', 'tied', 'diag']  # The covariance type

for cv_type in cv_types:
    
    for n_components in n_components_range:
        
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        gmm1 = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        gmm.fit(tf_idf_vectors.toarray())
        gmm1.fit(tf_idf1.toarray())
        bic = gmm.bic(tf_idf_vectors.toarray())  # get Model BIC
        bic1 = gmm1.bic(tf_idf1.toarray())
        if bic < lowest_bic:  # save the model with lowest BIC sofar
            lowest_bic = bic
            best_gmm = gmm
        if bic1 < lowest_bic1:
            lowest_bic1 = bic1
            best_gmm1 = gmm1


# In[71]:


lowest_bic
best_gmm


# In[72]:


lowest_bic1
best_gmm1


# In[73]:


clusterer = GaussianMixture(9)
    
cluster_labels = clusterer.fit_predict(tf_idf_vectors.toarray())

clusterer1 = GaussianMixture(9)

cluster1_labels = clusterer1.fit_predict(tf_idf1.toarray())


# In[74]:


df["labels"] = cluster_labels

idx_to_word = {values:key for key,values in vectorizer.vocabulary_.items()}

vocab_idx = np.argsort(clusterer.means_, axis=1)[:,::-1][:,:10]

for i in clusterer.weights_.argsort():
    print("="*50)
    print("Cluster ",i," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# In[75]:


f["labels"] = cluster1_labels

idx_to_word = {values:key for key,values in count1.vocabulary_.items()}

vocab_idx = np.argsort(clusterer1.means_, axis=1)[:,::-1][:,:10]


#picking  5 important clusters based on weights
for i in clusterer1.weights_.argsort()[::-1][:5]:
    print("="*50)
    print("Cluster ",i," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# ## GaussianMixture with Word Vectors

# In[76]:


from sklearn.mixture import GaussianMixture
from sklearn import mixture
# We use a small scope of grid search to find the best parameters of GMM

lowest_bic = np.infty   # initial BIC is set to infinity
best_gmm = None
lowest_bic1 = np.infty   # initial BIC is set to infinity
best_gmm1 = None
n_components_range = range(2,10)    # The number of clusters

cv_types = ['spherical', 'tied', 'diag']  # The covariance type

for cv_type in cv_types:
    
    for n_components in n_components_range:
        
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        gmm1 = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        gmm.fit(word_vectors)
        gmm1.fit(word_vectors1)
        bic = gmm.bic(word_vectors)  # get Model BIC
        bic1 = gmm1.bic(word_vectors1)
        if bic < lowest_bic:  # save the model with lowest BIC sofar
            lowest_bic = bic
            best_gmm = gmm
        if bic1 < lowest_bic1:
            lowest_bic1 = bic1
            best_gmm1 = gmm1


# In[77]:


lowest_bic
best_gmm


# In[78]:


lowest_bic1
best_gmm1


# In[79]:


clusterer = GaussianMixture(9)
    
cluster_labels = clusterer.fit_predict(word_vectors)

clusterer1 = GaussianMixture(4)

cluster1_labels = clusterer1.fit_predict(word_vectors1)


# In[80]:


clusterer = GaussianMixture(9)
    
cluster_labels = clusterer.fit_predict(word_vectors)

df["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in clusterer.weights_.argsort()[::-1][:5]:
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[81]:


clusterer = GaussianMixture(4)
    
cluster_labels = clusterer.fit_predict(word_vectors1)

f["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in clusterer.weights_.argsort()[::-1][:5]:
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# ## Agglomerative Clustering

# ### Agglomerative Clustering with Bag of words

# In[84]:


# Elbow Method for Heirarchical Clustering

from sklearn.cluster import MiniBatchKMeans
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

# define dataset
# define the model

print("This is for Indian news")
model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(bow_vectors.toarray())        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[89]:



from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(10)
    
cluster_labels = clusterer.fit_predict(bow_vectors.toarray())

df["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in range(10):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[86]:


# Elbow Method for Heirarchical Clustering

from sklearn.cluster import MiniBatchKMeans
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

# define dataset
# define the model

print("This is for American news")
model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(bow1.toarray())        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[88]:


from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(10)
    
cluster_labels = clusterer.fit_predict(bow1.toarray())

f["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in range(10):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(f[f["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ### Agglomerative Clustering with TFIDF

# In[90]:


# Elbow Method for Heirarchical Clustering

from sklearn.cluster import MiniBatchKMeans
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

# define dataset
# define the model

print("This is for Indian news")
model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(tf_idf_vectors.toarray())        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[91]:


from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(9)
    
cluster_labels = clusterer.fit_predict(tf_idf_vectors.toarray())

df["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in range(9):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[93]:


# Elbow Method for Heirarchical Clustering

from sklearn.cluster import MiniBatchKMeans
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

# define dataset
# define the model

print("This is for American news")
model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(tf_idf1.toarray())        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[106]:


#We approximate the elbow at 10 as the visualiser does not give the perfect elbow always , it has to be decided by us
from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(10)
    
cluster_labels = clusterer.fit_predict(tf_idf1.toarray())

f["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in range(10):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(f[f["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# ### Agglomerative Clustering with Word Vectors

# In[98]:


# Elbow Method for Heirarchical Clustering

from sklearn.cluster import MiniBatchKMeans
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

# define dataset
# define the model

print("This is for Indian news")
model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(word_vectors)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[99]:


from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(9)
    
cluster_labels = clusterer.fit_predict(word_vectors)

df["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in range(8):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[102]:



from sklearn.cluster import MiniBatchKMeans
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

# define dataset
# define the model

print("This is for American news")
model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(word_vectors1)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[104]:


from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(9)
    
cluster_labels = clusterer.fit_predict(word_vectors1)

f["labels"] = cluster_labels

#picking  5 important clusters based on weights
for i in range(9):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(f[f["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# ## DBSCAN

# ### DBSCAN with Bag of words

# In[111]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
neighbors = NearestNeighbors(n_neighbors=550)
neighbors1 = NearestNeighbors(n_neighbors=550)
neighbors_fit = neighbors.fit(bow_vectors)
distances, indices = neighbors_fit.kneighbors(bow_vectors)


# In[112]:


neighbors1_fit = neighbors1.fit(bow1)
distances1, indices1 = neighbors_fit.kneighbors(bow1)


# In[115]:


# Plotting K-distance Graph
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for Indian news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[116]:


distances1 = np.sort(distances1, axis=0)
distances1 = distances1[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for American news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[ ]:


CLUSTERING FOR INDIAN NEWS


# In[127]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(bow_vectors, metric='cosine')
#Due to high dimensionality of data we use cosine similarity distance as euclidian distance woud be a poor choice.


# In[143]:


from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.36, min_samples=3,metric='precomputed')


# In[144]:


cluster_labels = clusterer.fit_predict(distance_array)
df["labels"] = cluster_labels
for i in sorted(list(set(cluster_labels))):
    
    if i== -1:
        continue
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()

FOR AMERICAN NEWS
# In[145]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(bow1, metric='cosine')
#Due to high dimensionality of dta we use cosine similarity distance as euclidian distance woud be a poor choice.
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.3, min_samples=3,metric='precomputed')


# In[146]:


cluster_labels = clusterer.fit_predict(distance_array)
f["labels"] = cluster_labels
for i in sorted(list(set(cluster_labels))):
    
    if i== -1 :
        continue
    print("="*50)
    if i == 6:
        break
    print("Cluster ",i+1," talks about :")
    print()
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# ### DBSCAN with TFIDF

# In[306]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
neighbors = NearestNeighbors(n_neighbors=550)
neighbors1 = NearestNeighbors(n_neighbors=550)
neighbors_fit = neighbors.fit(tf_idf_vectors)
distances, indices = neighbors_fit.kneighbors(tf_idf_vectors)


# In[307]:


neighbors1_fit = neighbors1.fit(tf_idf1)
distances1, indices1 = neighbors_fit.kneighbors(tf_idf1)


# In[308]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for Indian news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[309]:


distances1 = np.sort(distances1, axis=0)
distances1 = distances1[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for American news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[314]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(tf_idf_vectors, metric='cosine')
#Due to high dimensionality of dta we use cosine similarity distance as euclidian distance woud be a poor choice.
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.7, min_samples=3,metric='precomputed')


# In[315]:


cluster_labels = clusterer.fit_predict(distance_array)
df["labels"] = cluster_labels
for i in sorted(list(set(cluster_labels))):
    
    if i== -1 :
        continue
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[322]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(tf_idf1, metric='cosine')
#Due to high dimensionality of dta we use cosine similarity distance as euclidian distance woud be a poor choice.
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.62, min_samples=3,metric='precomputed')


# In[323]:


cluster_labels = clusterer.fit_predict(distance_array)
f["labels"] = cluster_labels
for i in sorted(list(set(cluster_labels))):
    
    if i== -1 :
        continue
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# # DBSCAN WITH WORD2VEC

# In[147]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
neighbors = NearestNeighbors(n_neighbors=550)
neighbors1 = NearestNeighbors(n_neighbors=550)
neighbors_fit = neighbors.fit(word_vectors)
distances, indices = neighbors_fit.kneighbors(word_vectors)


# In[148]:


neighbors1_fit = neighbors1.fit(word_vectors1)
distances1, indices1 = neighbors_fit.kneighbors(word_vectors1)


# In[149]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for Indian news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[150]:


distances1 = np.sort(distances1, axis=0)
distances1 = distances1[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for American news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()

CLUSTERING FOR INDIAN NEWS
# In[182]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(word_vectors, metric='cosine')
#To prevent any dimension problem and increase accuracy we use cosine instead of euclidean distance
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.47, min_samples=3,metric='precomputed')


# In[183]:


cluster_labels = clusterer.fit_predict(distance_array)
df["labels"] = cluster_labels
for i in sorted(list(set(cluster_labels))):
    
    if i== -1 :
        continue
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()

CLUSTERING FOR AMERICAN NEWS
# In[187]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(word_vectors1, metric='cosine')
#To prevent any dimension problem and increase accuracy we use cosine instead of euclidean distance
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.5, min_samples=4,metric='precomputed')


# In[188]:


cluster_labels = clusterer.fit_predict(distance_array)
f["labels"] = cluster_labels
for i in sorted(list(set(cluster_labels))):
    
    if i== -1 :
        continue
    if i == 1:
        break
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# ## LDA

# In[359]:


from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.model_selection import GridSearchCV

search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

lda = LDA(learning_method='online', learning_offset=50.,random_state=100)
lda1 = LDA(learning_method='online', learning_offset=50.,random_state=100)
model = GridSearchCV(lda, param_grid=search_params)
model1 = GridSearchCV(lda1, param_grid=search_params)

model.fit(tf_idf_vectors.toarray())
model1.fit(tf_idf1.toarray())

best_lda_model = model.best_estimator_
best_lda_model1 = model1.best_estimator_

print("Best Model's Params: ", model.best_params_)

print("Best Log Likelihood Score: ", model.best_score_)


print("Model Perplexity: ", best_lda_model.perplexity(tf_idf_vectors.toarray()))


print("Best second Model's Params: ", model1.best_params_)

print("Best Log Likelihood Score: ", model1.best_score_)


print("Model Perplexity: ", best_lda_model1.perplexity(tf_idf1.toarray()))


# In[329]:


lda_output = best_lda_model.transform(tf_idf_vectors)

topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

docnames = ["Doc" + str(i) for i in range(len(df.text))]

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)


dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic["dominant_topic"] = dominant_topic


def color_green(val):
    color = "green" if val > .1 else "black"
    return "color: {col}".format(col=color)
def make_bold(val):
    weight = 700 if val > .1 else 400
    return "font-weight: {weight}".format(weight=weight)

df_document_topics = df_document_topic.head(5).style.applymap(color_green).applymap(make_bold)
df_document_topics


# In[337]:


lda_output1 = best_lda_model1.transform(tf_idf1)

topicnames = ["Topic" + str(i) for i in range(best_lda_model1.n_components)]

docnames = ["Doc" + str(i) for i in range(len(f.text))]

df_document_topic = pd.DataFrame(np.round(lda_output1, 2), columns=topicnames, index=docnames)


dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic["dominant_topic"] = dominant_topic


def color_green(val):
    color = "green" if val > .1 else "black"
    return "color: {col}".format(col=color)
def make_bold(val):
    weight = 700 if val > .1 else 400
    return "font-weight: {weight}".format(weight=weight)

df_document_topics1 = df_document_topic.head(5).style.applymap(color_green).applymap(make_bold)
df_document_topics1


# In[344]:


df_topic_keywords = pd.DataFrame(best_lda_model.components_)
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
df_topic_keywords.head()


# In[345]:


df_topic_keywords = pd.DataFrame(best_lda_model1.components_)
df_topic_keywords.columns = vectorizer1.get_feature_names()
df_topic_keywords.index = topicnames
df_topic_keywords.head()


# * Below we are showing top 15 keywords for each topic

# In[353]:


def show_topics(vectorizer=tf_idf_vectors,lda_model=best_lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=count, lda_model=best_lda_model, n_words=15)

df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Important Word '+str(i+1) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i+1) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


# In[358]:


def show_topics(vectorizer=tf_idf1,lda_model=best_lda_model, n_words=20):
    keywords = np.array(vectorizer1.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=count, lda_model=best_lda_model1, n_words=15)

df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Important Word '+str(i+1) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i+1) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


# # VISUALISATION OF BOTH DATA'S WORD FREQUENCIES AND POLARITY OF TEXT USING TEXTBLOB

# In[365]:


li = df["text"].to_list()
#Tokenize to list of words for every sentence that has been cleaned for news title
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
tokenized_sents_title = [word_tokenize(i) for i in li]
tokenized_sents_title


# In[366]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
str_words = ' '.join(li)
wc = WordCloud(background_color = 'white',max_words = 2000)
wc.generate(str_words)
plt.imshow(wc)


# In[23]:


from textblob import TextBlob

sentiment_objects = [TextBlob(t) for t in df.text]
sentiment_values = [[t.sentiment.polarity, str(t)] for t in sentiment_objects]
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "text"])
sentiment_df.head()


# In[32]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from news articles on Climate Change for Indian news")
plt.show()


# In[29]:


# Remove polarity values equal to zero
sentiment_df = sentiment_df[sentiment_df.polarity != 0]


# In[31]:


fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with break at zero
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from news articles on Climate Change for Indian news")
plt.show()


# In[33]:


sentiment_objects = [TextBlob(t) for t in f.text]
sentiment_values = [[t.sentiment.polarity, str(t)] for t in sentiment_objects]
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "text"])
sentiment_df.head()


# In[37]:


fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from news articles on Climate Change for American news")
plt.show()


# In[38]:


# Remove polarity values equal to zero
sentiment_df = sentiment_df[sentiment_df.polarity != 0]


# In[39]:


fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with break at zero
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from news articles on Climate Change for American news")
plt.show()


# # CONCLUSION ANALYSIS
First we see what is the average clusters for Kmeans with different vectorizers for the Indian news.SO we get 8 , 9 and 4 clusters for bag of words, TF-IDF and WordVectors respectively.So we decide to take an approximate average of 8 
clusters for the Kmeans algorithm for Indian news.For american news the same case of Kmeans with different vectorizers is giving results as 8 , 6 and 4 for bag of words, TF-IDF and WordVectors respectively.So we take an approximation of 6 clusters for American news by the mean of these values.First we see what is the average clusters for GMM with different vectorizers for the Indian news.SO we get 9 , 9 and 9 clusters for bag of words, TF-IDF and WordVectors respectively.So we decide to take an approximate average of 9 
clusters for the GMM algorithm for Indian news.For american news the same case of GMM with different vectorizers is giving results as 7 , 9 and 4 for bag of words, TF-IDF and WordVectors respectively.So we take an approximation of 7 clusters for American news by the mean of these values.First we see what is the average clusters for Aglomerative clustering with different vectorizers for the Indian news.SO we get 10 , 9 and 9 clusters for bag of words, TF-IDF and WordVectors respectively.So we decide to take an approximate average of 9 
clusters for the Aglomerative algorithm for Indian news.For american news the same case of GMM with different vectorizers is giving results as 10 , 10 and 9 for bag of words, TF-IDF and WordVectors respectively.So we take an approximation of 10 clusters for American news by the mean of these values.First we see what is the average clusters for Aglomerative clustering with different vectorizers for the Indian news.SO we get 10 , 9 and 9 clusters for bag of words, TF-IDF and WordVectors respectively.So we decide to take an approximate average of 9 
clusters for the Aglomerative algorithm for Indian news.For american news the same case of GMM with different vectorizers is giving results as 10 , 10 and 9 for bag of words, TF-IDF and WordVectors respectively.So we take an approximation of 10 clusters for American news by the mean of these values.We used eps=0.36,eps= 0.7,eps=0.47 for bag of words, TF-IDF and WordVectors respectively in case of Indian news.    
We used eps=0.3 ,eps=0.62 ,eps=0.5 for bag of words, TF-IDF and WordVectors respectively in case of American news.
First we see what is the average clusters for Aglomerative clustering with different vectorizers for the Indian news.so we get 6 , 5 and 1 clusters for bag of words, TF-IDF and WordVectors respectively.So we decide to take an approximate median average of 5 
clusters for the Aglomerative algorithm for Indian news.For american news the same case of GMM with different vectorizers is giving results as 6 , 5 and 1 for bag of words, TF-IDF and WordVectors respectively.So we take an approximation of  5 for American news by the median of these values.

So for finally evaluating metric using BERT we can expect an approximate eps of 0.51 in case of Indian news and 0.47 in case of American news.
# In[ ]:




