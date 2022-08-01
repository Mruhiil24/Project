#!/usr/bin/env python
# coding: utf-8

# In[153]:


get_ipython().system('pip install webdriver_manager')
get_ipython().system('pip install selenium ')


# In[17]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import time
import itertools


# In[1]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import time
driver = webdriver.Chrome(ChromeDriverManager().install())


# ## Scrape NBC News

# In[138]:


driver.get("https://www.nbcnews.com/climate-in-crisis")


# In[139]:


import time
while True:
    
    try:
        next_button = driver.find_element_by_xpath("//div[@class='feeds__load-more-wrapper']")
        next_button.click()
    except:
        break;        
    time.sleep(2)   


# In[140]:




url_components = driver.find_elements_by_xpath("//div[@class='wide-tease-item__wrapper df flex-column flex-row-m flex-nowrap-m']/div/a")




# In[141]:


all_urls = [url.get_attribute("href") for url in url_components]

urls_to_scrape = [url for url in all_urls if "video" not in url]


# In[142]:


urls_to_scrape


# In[143]:


from tqdm import tqdm

article_texts = []

for url in tqdm(urls_to_scrape):
    
    driver.get(url)
    
    paragraphs_in_article =  driver.find_elements_by_xpath("//div[contains(@class,'article-body')]//p")[::-1]
    
    news_article_text = ""

    for paragraph in paragraphs_in_article:
        text = paragraph.text
    
        news_article_text = news_article_text + " " + text
        
    article_texts.append(news_article_text)


# In[144]:


import pandas as pd

nbc_news = pd.DataFrame(article_texts,columns=["text"])


# In[145]:


nbc_news.to_csv("nbc_news.csv",index=False)


# In[150]:


nbc_news.tail()


# # Scrape New York Times

# In[173]:


driver.get("https://www.nytimes.com/section/climate")


# In[174]:


driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(1)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(1)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(1)


# In[188]:


urls = driver.find_elements_by_xpath("//li[contains(@class,'css-ye6x8s')]//a")


# In[191]:


urls_to_scrape = [url.get_attribute("href") for url in urls]


# In[192]:


urls_to_scrape


# In[ ]:


for url in urls_to_scrape[:5]:
    driver.get(url)


# In[ ]:





# In[197]:


article_texts = []

for url in tqdm(urls_to_scrape):
    
    driver.get(url)
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    paragarhs = driver.find_elements_by_xpath("//section[contains(@name,'articleBody')]//p")
    
    article_text = ""

    for paragraph in paragarhs:
        
        text = paragraph.text
    
        article_text = article_text + " " + text
        
    article_texts.append(news_article_text)


# In[198]:


article_texts[-1]


# In[ ]:


ny_times = pd.DataFrame(article_texts,columns=["text"])


# In[ ]:


nbc_news.to_csv("ny_times.csv",index=False)


# In[ ]:





# # Scrape Hindustan Times

# In[28]:


driver.get("https://www.hindustantimes.com/topic/climate-change/page-1")


# In[37]:


from tqdm import tqdm
import time

urls_to_scrape = []

for page_no in tqdm(range(1,35)):
    
    driver.get(f"https://www.hindustantimes.com/topic/climate-change/page-{page_no}")
    
    time.sleep(2)
    
    urls_in_page = driver.find_elements_by_xpath("//section[contains(@data-url,'/topic/climate-change')]//h3[contains(@class,'hdg3')]/a")
    
    urls_in_page = [url.get_attribute("href") for url in urls_in_page]
    
    urls_to_scrape.append(urls_in_page)
    


# In[38]:


urls_to_scrape = list(itertools.chain(*urls_to_scrape))


# In[39]:


print("Total news articles related to climate to be scrapped from Hindustan times :", len(urls_to_scrape))


# In[51]:


urls_to_scrape[:34]


# In[48]:


from tqdm import tqdm

article_texts = []

for url in tqdm(urls_to_scrape):
    
    driver.get(url)
    
    paragraphs=  driver.find_elements_by_xpath("//div[contains(@class,'storyDetails')]//p")[:-2]
    
    article_text = ""

    for p in paragraphs:
        text = p.text
    
        article_text = article_text + " " + text
        
    article_texts.append(article_text)


# In[ ]:




