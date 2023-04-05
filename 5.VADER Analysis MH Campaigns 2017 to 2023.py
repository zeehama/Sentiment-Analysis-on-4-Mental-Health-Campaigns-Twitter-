#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import IPython
import re
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer

import string
import re

import wordcloud as WordCloud
from wordcloud import WordCloud

from textblob import TextBlob
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONSMH_ca


# In[2]:


import sys
sys.path.insert(0, '../../_functions_')

from functions_EDA import *
from fx_NLP import *


# In[3]:


df = pd.read_csv('../datasets/MH_Campaign_Tweets_Tokenised_1723.csv')


# In[4]:


df_copy =df.copy()


# In[5]:


df.info()


# In[14]:


df.iloc[310815,:].tweet


# In[15]:


df.iloc[310815,:].processed_tweet


# In[16]:


df.iloc[310815,:].tweet_emoji_punc


# In[17]:


#writing function to see different versions of one tweet, to understand how they look once cleaned
def get_tweet_versions(df,index):
    original_tweet=df.iloc[index,:].tweet
    processed_tweet=df.iloc[index,:].processed_tweet
    tokenised_tweet=df.iloc[index,:].tokenised_tweet
    tweet_emoji_punc=df.iloc[index,:].tweet_emoji_punc
    
    return{'original_tweet': original_tweet,
           'Tweet without stopwords, punctuation or emojis': processed_tweet,
           'tokenised_tweet': tokenised_tweet,
           'tweet with punctuation and emojis': tweet_emoji_punc}


# In[12]:


get_tweet_versions(df,56789)


# # VADER Sentiment Analysis

# In[ ]:


#!pip install vaderSentiment


# In[ ]:


# load the SentimentIntensityAnalyser object in
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


# In[ ]:


# Define a function to calculate the sentiment scores for each tweet
def get_sentiment_scores(text):
    scores = analyzer.polarity_scores(text)
    return scores


# In[ ]:


# Apply the sentiment analysis function to the tweet column
df['sentiment_scores'] = df['tweet'].apply(get_sentiment_scores)


# In[ ]:


# Extract the compound scores from the sentiment_scores column
df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])


# In[ ]:


# save the new dataset that now has the compound scores included
df.to_csv('MH_Campaign_Tweets_Sentiment_Scored_1723.csv')


# # Load df w/processed tweet & compound score from here

# In[3]:


df = pd.read_csv('../datasets/MH_Campaign_Tweets_Sentiment_Scored_1723.csv')


# In[5]:


# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot the sentiment scores by campaign using a bar chart with custom color
df.groupby('campaign')['compound_score'].mean().plot(kind='bar', color='#72C4C0')
plt.xlabel('Campaign')
plt.ylabel('Sentiment Score')
plt.title('Average Sentiment Scores by Campaign')
plt.show()


# In[6]:


# Plot the sentiment scores by year using a line chart
df.groupby(df['year'])['compound_score'].mean().plot(kind='bar',color='#97288F')
plt.xlabel('Year')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Scores for EDAW Tweets by Year')
plt.show()


# In[7]:


#exploring the distribution of the sentiment score for Mental Health Awareness Week and Eating Disorders Awareness Week 
import matplotlib.pyplot as plt
import seaborn as sns

df = df[(df['campaign'] == 'EDAW') | (df['campaign'] == 'MHAW')]

# Filter the DataFrame by the campaign
#df = df[df['campaign'] == 'EDAW']

# Create a boxplot of sentiment scores by campaign and year
sns.boxplot(x='campaign', y='compound_score', hue='year', data=df)

# Set the title and labels for the plot
plt.title('Sentiment Scores by Campaign and Year')
plt.xlabel('Campaign')
plt.ylabel('Sentiment Score')

# Move the legend to the right-hand side of the chart
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

# Show the plot


# In[41]:


def pos_neg_wc(df, campaign, year= None , custom_stopwords=[]):
     # filter dataframe by campaign and year if year is not None
    if year is not None:
        df = df[(df['campaign'] == campaign) & (df['year'] == year)]
    else:
        df = df[df['campaign'] == campaign]

    #sort scores in order of low to high
    df = df.sort_values(by='compound_score')

    # Extract the top N positive and negative tweets
    top_positive_tweets = df[df['compound_score'] > 0]['processed_tweet'][-2000:]
    top_negative_tweets = df[df['compound_score'] < 0]['processed_tweet'][:2000]

    # Join the tweets together into a single string for each group
    positive_tweets_str = ' '.join(top_positive_tweets)
    negative_tweets_str = ' '.join(top_negative_tweets)

    # Set up stopwords
    #nltk.download('stopwords')
    
    
    stopwords = nltk.corpus.stopwords.words('english')
    if custom_stopwords is not None:
        stopwords.extend(custom_stopwords)
        
    #Define a list of colors to use
    colors = ["#97288F","#72C4C0","#402248"]

    def get_word_color(word, *args, **kwargs):
        # Use a hash function to map the word to a color from the list of colors
        hash_value = hash(word)
        color_index = hash_value % len(colors)
        return colors[color_index]  

    # Set the desired image width and height
    #using width and height of 1920 and 1080 to create a 1080p image
    width, height = 1920, 1080
    
    # Create a WordCloud object with the given parameters
    wordcloud = WordCloud(width=width, height=height, background_color='white', stopwords=stopwords,
                          colormap='cool', color_func=get_word_color).generate(positive_tweets_str)
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Top 2000 Most Positive {campaign} Tweets {year}')
    plt.show()

    # Create a word cloud for the top negative tweets
    wordcloud = WordCloud(width=width, height=height, background_color='white', stopwords=stopwords,
                          colormap='cool', color_func=get_word_color).generate(negative_tweets_str)
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Top 2000 Most Negative {campaign} Tweets {year}')
    plt.show()
    


# In[44]:


pos_neg_wc(df, 'EDAW', year=None, custom_stopwords=['u','edaw','support','disorders','beated','thank','nedawareness',
                                                    'edrecovery','EatingDisorderAwarenessWeek','disordereating','help',
                                                    'recovery','Recovery','Hope','hope','anorexia','bulima','binge','nervosa',
                                                    'https','eatingdisorder','eating','disorder','amp','t','co',
                                  'EatingDisordersAwarenessWeek','awareness','week','s','ED','eatingdisorders',
                                         'MentalHealthAwarenessWeek', 'Mental','Health','mentalhealth','national','people',
                                                   'bulimia'])


# # Investigating distribution of all scores

# In[45]:


df['compound_score'].value_counts()


# In[48]:


EDAW=df[df['campaign']=='EDAW']
#now working with EDAW tweets only


# In[49]:


#dataframe with POSITIVE tweets from EDAW
df_pos = EDAW.loc[EDAW.compound_score >= 0.95]
df_pos

# only corpus of POSITIVE comments
pos_tweets = df_pos['tweet'].tolist()
pos_tweets


# In[50]:


# dataframe with NEGATIVE tweets during EDAW
df_neg = EDAW.loc[EDAW.compound_score < 0.0]

# only corpus of NEGATIVE comments
neg_tweets = df_neg['tweet'].tolist()
neg_tweets


# In[51]:


#comparing length of both positive and negative tweets


# In[52]:


len(df_neg['tweet'])


# In[53]:


df_pos['tweet_length'] = df_pos['tweet'].apply(len)
df_neg['tweet_length'] = df_neg['tweet'].apply(len)


# In[54]:


df_neg['tweet_length']


# In[55]:


sns.set_style("whitegrid")
plt.figure(figsize=(8,5))

sns.distplot(df_pos['tweet_length'], kde=True, bins=50, color='green')
sns.distplot(df_neg['tweet_length'], kde=True, bins=50, color='coral')

plt.title('\nDistribution Plot for Length of Tweets\n')
plt.legend(['Positive Comments', 'Negative Comments'])
plt.xlabel('\nTweet Length')
plt.ylabel('Percentage of Tweets\n');


# The mode for the text length of positive comments can be found more to the right than for the negative comments, which means most of the positive comments are longer than most of the negative comments. However, there are two peaks for negative comments.

# In[56]:


# read some positive tweets
pos_tweets[10:15]


# In[57]:


# read some negative tweets
neg_tweets[10:15]


# # Frequency Distributions

# In[ ]:


#!pip install yellowbrick


# In[60]:


# importing libraries
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text.freqdist import FreqDistVisualizer
from yellowbrick.style import set_palette


# In[61]:


import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')
en_stopwords = stopwords.words('english')

# Define your custom list of stop words
custom_stopwords = ['recovery','one','get','ed','edaw','beated','eatingdisorderawarenessweek','people',
                    'eatingdisordersawarenessweek','health','mental','https', 'eatingdisorder', 'eating', 
                    'disorder', 'amp', 't', 'co', 'EatingDisordersAwarenessWeek', 'awareness', 'week', 's', 
                    'ED', 'eatingdisorders', 'MentalHealthAwarenessWeek', 'Mental', 'Health', 'mentalhealth', 
                    'EDAW2017', '0H4TYXsBai', 'beatED', 'EDAW2018', 'disorders', 'EatingDisorderAwarenessWeek', 
                    'EDAW2019', 'EDAW2020', 'EDAW', 'BH', 'beatED', 'EDAW2021', 'EDAW2023', 'National', 'NedAwareness',
                   'EDAW2022',"national","anorexia","edaw2017","edaw2018","edaw2019","binge","bulimia","edaw2022",
                    "nedawareness","edaw2023","uk"]

# Combine the two lists of stop words
stopwords = custom_stopwords + en_stopwords

# Vectorize the text, removing the stop words
vectorizer = CountVectorizer(stop_words=stopwords)
docs = vectorizer.fit_transform(neg_tweets)
features = vectorizer.get_feature_names()


# In[62]:


vectorizer.get_feature_names()


# In[63]:


visualizer = FreqDistVisualizer(features=features, n=30,color='#D80D2B',title='30 Most Frequent Words Used in Negative Sentiment Tweets Related to EDAW')
visualizer.fit(docs)

# Show the visualizer
visualizer.poof();


# In[64]:


docs = vectorizer.fit_transform(pos_tweets)
docs
features = vectorizer.get_feature_names()


# In[65]:


visualizer = FreqDistVisualizer(features=features, n=30, color='#72C4C0',title='30 Most Frequent Words Used in Positive Sentiment Tweets Related to EDAW')
visualizer.fit(docs)
visualizer.poof();


# # Plotting most used emojis

# In[66]:


#!pip install emot

#this is the code to define the function that will be extracting emojis
import emot
emot_obj = emot.emot()


# In[67]:


# Define a function to extract emoticons
def extract_emoticons(text):
  res = emot_obj.emoji(text)
  return res['value']


# In[68]:


# Filter the DataFrame by campaign and negative sentiment
EDAW_neg = df[(df['campaign'] == 'EDAW') & (df['compound_score'] < -0.5)]


# In[69]:


# Filter the DataFrame by campaign and positive sentiment
EDAW_pos = df[(df['campaign'] == 'EDAW') & (df['compound_score'] > 0.5)]


# In[70]:


# Filter the DataFrame by campaign and neutral sentiment
EDAW_neu = df[(df['campaign'] == 'EDAW') & (df['compound_score'] == 0)]


# In[71]:


#applying the function to your column with tweets
EDAW_neg['emoticons'] = EDAW_neg['tweet'].apply(extract_emoticons)


# In[72]:


#applying the function to your column with tweets
EDAW_pos['emoticons'] = EDAW_pos['tweet'].apply(extract_emoticons)


# In[73]:


#applying the function to your column with tweets
EDAW_neu['emoticons'] = EDAW_neu['tweet'].apply(extract_emoticons)


# In[78]:


#this part you need to count emojis
from collections import Counter
import collections
#code to count emojis
# Count the emojis in each list
EDAW_neg['emoticons'].apply(lambda x: collections.Counter(x))
EDAW_neg['emoticons'].apply(lambda x: collections.Counter(x))
EDAW_neu['emoticons'].apply(lambda x: collections.Counter(x))

# Combine the counts
combined_counts_neg = sum(EDAW_neg['emoticons'].apply(lambda x: collections.Counter(x)), collections.Counter())
combined_counts_pos = sum(EDAW_pos['emoticons'].apply(lambda x: collections.Counter(x)), collections.Counter())
combined_counts_neu = sum(EDAW_neu['emoticons'].apply(lambda x: collections.Counter(x)), collections.Counter())

# Transform it into a dict
emoji_dict_neg = dict(combined_counts_neg)
emoji_dict_pos = dict(combined_counts_pos)
emoji_dict_neu = dict(combined_counts_neu)

# Sort it in a descending order
sorted_emoji_dict_neg = dict(sorted(emoji_dict_neg.items(), key=lambda x: x[1], reverse=True))
sorted_emoji_dict_neu = dict(sorted(emoji_dict_neu.items(), key=lambda x: x[1], reverse=True))
sorted_emoji_dict_pos = dict(sorted(emoji_dict_pos.items(), key=lambda x: x[1], reverse=True))


# In[84]:


from plotly.subplots import make_subplots
# Keep the top 20
d = {k: v for i, (k, v) in enumerate(sorted_emoji_dict_neg.items()) if i < 20}
e = {k: v for i, (k, v) in enumerate(sorted_emoji_dict_pos.items()) if i < 20}
f = {k: v for i, (k, v) in enumerate(sorted_emoji_dict_neu.items()) if i < 20}

# Convert the dict to a DataFrame for Plotly
dfneg = pd.DataFrame(list(d.items()), columns=['Emojis', 'Count'])
dfpos = pd.DataFrame(list(e.items()), columns=['Emojis', 'Count'])
dfneu = pd.DataFrame(list(f.items()), columns=['Emojis', 'Count'])

#this is to display a dataframe with top 20
dfneg.groupby('Emojis').sum()['Count'].sort_values(ascending=False).reset_index()
dfpos.groupby('Emojis').sum()['Count'].sort_values(ascending=False).reset_index()
dfneu.groupby('Emojis').sum()['Count'].sort_values(ascending=False).reset_index()

# Create bar charts
fig = make_subplots(rows=1, cols=3, subplot_titles=('Negative Tweets', 'Positive Tweets', 'Neutral Tweets'))

data1 = [go.Bar(x=dfneg['Emojis'], y=dfneg['Count'])]
layout1 = go.Layout(title='20 Most Common emojis in negative EDAW tweets')
fig.add_trace(go.Bar(x=dfneg['Emojis'], y=dfneg['Count']), row=1, col=1)
fig.update_xaxes(title_text="Emojis",row=1, col=1)
fig.update_yaxes(title_text="Count",row=1, col=1)
fig.update_layout(layout1)

data2 = [go.Bar(x=dfpos['Emojis'], y=dfpos['Count'])]
layout2 = go.Layout(title='20 Most Common emojis in positive EDAW tweets')
fig.add_trace(go.Bar(x=dfpos['Emojis'], y=dfpos['Count']), row=1, col=2)
fig.update_xaxes(title_text="Emojis", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_layout(layout2)

data3 = [go.Bar(x=dfneu['Emojis'], y=dfneu['Count'])]
layout3 = go.Layout(title='20 Most Common emojis in neutral EDAW tweets')
fig.add_trace(go.Bar(x=dfneu['Emojis'], y=dfneu['Count']), row=1, col=3)
fig.update_xaxes(title_text="Emojis",row=1, col=3)
fig.update_yaxes(title_text="Count",row=1, col=3)
fig.update_layout(layout3)


# Display the chart
fig.show()


# In[85]:


#negative expanded
data = [go.Bar(x=dfneg['Emojis'], y=dfneg['Count'])]
layout = go.Layout(title='20 most common emojis in negative tweets')
fig = go.Figure(data=data, layout=layout)
# display the chart
iplot(fig)


# In[86]:


#positive expanded
data = [go.Bar(x=dfpos['Emojis'], y=dfpos['Count'])]
layout = go.Layout(title='20 most common emojis in positive tweets')
fig = go.Figure(data=data, layout=layout)
# display the chart
iplot(fig)

