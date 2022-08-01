#!/usr/bin/env python
# coding: utf-8

# ## Data Cleaning

# In[10]:


import string
import pandas as pd 
import numpy as np
df1 = pd.read_csv("nbc_news.csv")
df2 = pd.read_csv("ny_times.csv")
df3 = pd.read_csv("bbc_news_climate.csv")
df = pd.concat([df1, df2, df3])


# In[11]:


df.head()


# In[12]:


df = df.dropna()


# In[13]:


df = df[df["text"]!=""]


# In[14]:


df.shape


# In[15]:


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
    stopwords = nltk.corpus.stopwords.words('english') + ["said","climate","change","goverment","news"]   # remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i)>2])
    text= re.sub("\s[\s]+", " ",text).strip()            # remove repeated/leading/trailing spaces
    return text


# In[16]:


import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


# In[17]:


df.text = df.text.apply(clean_text).apply(lemmatize_text)
df.head()


# # Vectorization

# ## Bag of Words

# In[18]:


from sklearn.feature_extraction.text import  CountVectorizer
count = CountVectorizer(min_df=3, max_features=5000)
bow_vectors = count.fit_transform(df["text"].values)
#since there is no train and test, we are using fit_transform on all the data


# In[19]:


bow_vectors.shape


# ## TFIDF

# In[20]:


from sklearn.feature_extraction.text import  TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=3, max_features=1000)
tf_idf_vectors = vectorizer.fit_transform(df["text"].values)


# In[21]:


tf_idf_vectors.shape


# ## Word Vectors

# In[71]:


pip install wget


# In[3]:


get_ipython().system('wget https://s3.amazonaws.comhttps://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')
get_ipython().system('wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')


# In[4]:


import gzip
import shutil


# In[6]:


with gzip.open('GoogleNews-vectors-negative300.bin.gz', 'r') as f_in:
    with open('GoogleNews-vectors-negative300_.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[64]:


# pip install --upgrade gensim


# In[7]:


import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300_.bin', binary = True)


# In[8]:


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


# In[22]:


word_vectors = compute_WordVectors(df["text"].values,model)


# In[23]:


word_vectors.shape


# # Clustering

# ## Kmeans

# ## Kmeans with Bag of Words

# In[26]:


from sklearn.cluster import KMeans
n = list(range(2,40,2))
inertia = []
for i in n:
    
    clusterer = KMeans(i)
    
    clusterer.fit(bow_vectors)
    inertia.append(clusterer.inertia_)


# In[28]:


import matplotlib.pyplot as plt

plt.plot(n,inertia)
plt.title("KMeans with Bag of words for american news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[29]:


clusterer = KMeans(8)
    
cluster_labels = clusterer.fit_predict(bow_vectors)


# In[30]:


df["labels"] = cluster_labels
# f["labels"] = cluster1_labels 
idx_to_word = {values:key for key,values in count.vocabulary_.items()}

vocab_idx = np.argsort(clusterer.cluster_centers_, axis=1)[:,::-1][:,:10]


for i in range(8):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# In[32]:


cluster_labels.shape


# ## Kmeans with TFIDF Vectors

# In[ ]:





# In[33]:


from sklearn.cluster import KMeans
n = list(range(2,40,2))
inertia = []
for i in n:
    
    clusterer = KMeans(i)
 
    clusterer.fit(tf_idf_vectors)

    inertia.append(clusterer.inertia_)


# In[34]:


import matplotlib.pyplot as plt

plt.plot(n,inertia)
plt.title("KMeans with TFIDF for american news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[35]:


clusterer = KMeans(9)
    
cluster_labels = clusterer.fit_predict(tf_idf_vectors)


# In[36]:


df["labels"] = cluster_labels


# In[37]:


df.head()


# In[38]:


idx_to_word = {values:key for key,values in vectorizer.vocabulary_.items()}


# In[39]:


vocab_idx = np.argsort(clusterer.cluster_centers_, axis=1)[:,::-1][:,:10]


# In[40]:


for i in range(9):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# ## Kmeans with Word Vectors

# In[ ]:





# In[42]:


from sklearn.cluster import KMeans
n = list(range(2,40,2))
inertia = []

for i in n:
    
    clusterer = KMeans(i)

    clusterer.fit(word_vectors)
    
    inertia.append(clusterer.inertia_)
    


# In[43]:


plt.plot(n,inertia)
plt.title("KMeans with Word Vectors for american news")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


# In[44]:


clusterer = KMeans(4)
    
cluster_labels = clusterer.fit_predict(word_vectors)


# In[45]:


df["labels"] = cluster_labels


# In[46]:


df.head()


# In[47]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update(["country", "year", "state", "world", "time","one","area","people","que","los","la","para","una",
"por","del","con","dijo","como"])


# In[48]:


for i in range(4):
    print("="*50)
    print("Cluster ",i+1," talks about :")
    print()
    
    str_words = ' '.join(df[df["labels"]==i]["text"].values.tolist())
    wc = WordCloud(stopwords=stopwords,background_color = 'white',max_words = 100)
    wc.generate(str_words)
    plt.imshow(wc)
    plt.show()


# In[ ]:





# ## GaussianMixture

# ## GaussianMixture with Bag of Words

# In[49]:


from sklearn.mixture import GaussianMixture
from sklearn import mixture
# We use a small scope of grid search to find the best parameters of GMM

lowest_bic = np.infty   # initial BIC is set to infinity
best_gmm = None
n_components_range = range(2,10)    # The number of clusters

cv_types = ['spherical', 'tied', 'diag']  # The covariance type

for cv_type in cv_types:
    
    for n_components in n_components_range:
        
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        
        gmm.fit(bow_vectors.toarray())
        
        bic = gmm.bic(bow_vectors.toarray())  # get Model BIC
       
        if bic < lowest_bic:  # save the model with lowest BIC sofar
            lowest_bic = bic
            best_gmm = gmm
       


# In[50]:


lowest_bic
best_gmm


# In[51]:


clusterer = GaussianMixture(9)
    
cluster_labels = clusterer.fit_predict(bow_vectors.toarray())


# In[52]:


clusterer.weights_.argsort()[::-1][:5]


# In[53]:


df["labels"] = cluster_labels

idx_to_word = {values:key for key,values in count.vocabulary_.items()}

vocab_idx = np.argsort(clusterer.means_, axis=1)[:,::-1][:,:10]


#picking  5 important clusters based on weights
for i in clusterer.weights_.argsort()[::-1][:5]:
    print("="*50)
    print("Cluster ",i," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# ### GaussianMixture with TFIDF

# In[54]:


from sklearn.mixture import GaussianMixture
from sklearn import mixture
# We use a small scope of grid search to find the best parameters of GMM

lowest_bic = np.infty   # initial BIC is set to infinity
best_gmm = None

n_components_range = range(2,10)    # The number of clusters

cv_types = ['spherical', 'tied', 'diag']  # The covariance type

for cv_type in cv_types:
    
    for n_components in n_components_range:
        
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
       
        gmm.fit(tf_idf_vectors.toarray())
       
        bic = gmm.bic(tf_idf_vectors.toarray())  # get Model BIC
       
        if bic < lowest_bic:  # save the model with lowest BIC sofar
            lowest_bic = bic
            best_gmm = gmm
   


# In[55]:


lowest_bic
best_gmm


# In[56]:


clusterer = GaussianMixture(9)
    
cluster_labels = clusterer.fit_predict(tf_idf_vectors.toarray())


# In[57]:


df["labels"] = cluster_labels

idx_to_word = {values:key for key,values in vectorizer.vocabulary_.items()}

vocab_idx = np.argsort(clusterer.means_, axis=1)[:,::-1][:,:10]

for i in clusterer.weights_.argsort():
    print("="*50)
    print("Cluster ",i," talks about :")
    print()
    
    a = [print(idx_to_word[x]) for x in vocab_idx[i]]


# ## GaussianMixture with Word Vectors

# In[58]:


from sklearn.mixture import GaussianMixture
from sklearn import mixture
# We use a small scope of grid search to find the best parameters of GMM

lowest_bic = np.infty   # initial BIC is set to infinity
best_gmm = None

n_components_range = range(2,10)    # The number of clusters

cv_types = ['spherical', 'tied', 'diag']  # The covariance type

for cv_type in cv_types:
    
    for n_components in n_components_range:
        
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                    covariance_type=cv_type, random_state=42)
        
        gmm.fit(word_vectors)
     
        bic = gmm.bic(word_vectors)  # get Model BIC
   
        if bic < lowest_bic:  # save the model with lowest BIC sofar
            lowest_bic = bic
            best_gmm = gmm
     


# In[59]:


lowest_bic
best_gmm


# In[60]:


clusterer = GaussianMixture(9)
    
cluster_labels = clusterer.fit_predict(word_vectors)


# In[61]:


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


# ## Agglomerative Clustering

# ### Agglomerative Clustering with Bag of words

# In[66]:


pip install yellowbrick


# In[69]:


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
visualizer.fit(
    bow_vectors.toarray())        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[70]:


# Elbow Method for Heirarchical Clustering
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


# ### Agglomerative Clustering with TFIDF

# In[71]:


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
visualizer.fit(tf_idf_vectors.toarray())        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[75]:


#We approximate the elbow at 10 as the visualiser does not give the perfect elbow always , it has to be decided by us
from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(10)
    
cluster_labels = clusterer.fit_predict(tf_idf_vectors.toarray())

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


# ### Agglomerative Clustering with Word Vectors

# In[76]:



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
visualizer.fit(word_vectors)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[79]:


from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(9)
    
cluster_labels = clusterer.fit_predict(word_vectors)

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


# ## DBSCAN

# ### DBSCAN with Bag of words

# In[81]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
neighbors = NearestNeighbors(n_neighbors=550)

neighbors_fit = neighbors.fit(bow_vectors)
distances, indices = neighbors_fit.kneighbors(bow_vectors)


# In[82]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for American news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# FOR AMERICAN NEWS

# In[84]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(bow_vectors, metric='cosine')
#Due to high dimensionality of data we use cosine similarity distance as euclidian distance woud be a poor choice.


# In[85]:


from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.36, min_samples=3,metric='precomputed')


# In[86]:


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


# ### DBSCAN with TFIDF

# In[87]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
neighbors = NearestNeighbors(n_neighbors=550)
neighbors1 = NearestNeighbors(n_neighbors=550)
neighbors_fit = neighbors.fit(tf_idf_vectors)
distances, indices = neighbors_fit.kneighbors(tf_idf_vectors)


# In[88]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for Indian news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[89]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(tf_idf_vectors, metric='cosine')
#Due to high dimensionality of dta we use cosine similarity distance as euclidian distance woud be a poor choice.
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.7, min_samples=3,metric='precomputed')


# In[90]:


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


# # DBSCAN WITH WORD2VEC

# In[95]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
neighbors = NearestNeighbors(n_neighbors=550)

neighbors_fit = neighbors.fit(word_vectors)
distances, indices = neighbors_fit.kneighbors(word_vectors)


# In[96]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph for American news',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()

CLUSTERING FOR AMERICAN NEWS
# In[97]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
distance_array = pairwise_distances(word_vectors, metric='cosine')
#To prevent any dimension problem and increase accuracy we use cosine instead of euclidean distance
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.47, min_samples=3,metric='precomputed')


# In[98]:


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


# ## LDA

# In[100]:


from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.model_selection import GridSearchCV

search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

lda = LDA(learning_method='online', learning_offset=50.,random_state=100)
model = GridSearchCV(lda, param_grid=search_params)


model.fit(tf_idf_vectors.toarray())


best_lda_model = model.best_estimator_


print("Best Model's Params: ", model.best_params_)

print("Best Log Likelihood Score: ", model.best_score_)


print("Model Perplexity: ", best_lda_model.perplexity(tf_idf_vectors.toarray()))


# In[101]:


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


# In[102]:


df_topic_keywords = pd.DataFrame(best_lda_model.components_)
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
df_topic_keywords.head()


# * Below we are showing top 15 keywords for each topic

# In[103]:


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


# # VISUALISATION OF BOTH DATA'S WORD FREQUENCIES AND POLARITY OF TEXT USING TEXTBLOB

# In[104]:


li = df["text"].to_list()
#Tokenize to list of words for every sentence that has been cleaned for news title
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
tokenized_sents_title = [word_tokenize(i) for i in li]
tokenized_sents_title


# In[106]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
str_words = ' '.join(li)
wc = WordCloud(background_color = 'white',max_words = 2000)
wc.generate(str_words)
plt.imshow(wc)


# In[109]:


pip install textblob


# In[110]:


from textblob import TextBlob

sentiment_objects = [TextBlob(t) for t in df.text]
sentiment_values = [[t.sentiment.polarity, str(t)] for t in sentiment_objects]
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "text"])
sentiment_df.head()


# In[111]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from news articles on Climate Change for Indian news")
plt.show()


# In[112]:


# Remove polarity values equal to zero
sentiment_df = sentiment_df[sentiment_df.polarity != 0]


# In[113]:


fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with break at zero
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from news articles on Climate Change for Indian news")
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




