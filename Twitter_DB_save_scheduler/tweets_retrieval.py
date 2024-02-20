import tweepy
import configparser
import json
from datetime import date
from datetime import datetime
import re
from nltk.tokenize import TweetTokenizer
import re
import pandas as pd
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt  
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython.display import display
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
import numpy as np
import gensim.downloader as gapi
from nltk import word_tokenize
from collections import Counter
from typing import List
import os
import configparser
import os
from os import path
from pathlib import Path
import statistics
from pickle import load
#from flask_cors import CORS
from configobj import ConfigObj
from flask import Flask, request, jsonify, json, render_template,redirect, url_for
import mysql.connector as mySQL
import mysql.connector.errors
import schedule
import time
#####AUTO TO SCRIPT EINAI GIA THN DHMIOURGIA CLUSTER ME VASI DEDOMENWN. EINAI MEXRI STIGMIS EINAI H TELIKH EKDOSH 21/08/2023

stop = set(stopwords.words("english"))

import spacy
import nltk
#nltk.download('punkt')
#nlp = spacy.load('en_core_web_sm')
config = configparser.ConfigParser()
config.read('configuration.ini')

#nltk.download('punkt')
API_KEY = config['AuthenticationParams']['api_key']
API_SECRET = config['AuthenticationParams']['api_secret']
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAW7pwEAAAAApjTeepaQn2X9wPOU1UmPcgtX%2FP8%3DSCxYSDQsn8sURPwVwBVcSIroCiVsj4oJDMsm3yGGyI5EPtfVQq'#config['AuthenticationParams']["bearer_token"]
ACCESS_TOKEN = config['AuthenticationParams']["access_token"]
ACCESS_SECRET = config['AuthenticationParams']["access_secret"]

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
##auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
##auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)


##api = tweepy.API(auth, wait_on_rate_limit=True)


# We use Tweepy's OAuthHandler method to authenticate our credentials:
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)

# Then, we set our access tokens by calling the auth object directly:
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

# Finally, we can initialize the Twitter API. 
# 3NOTE: we will be using this `api` object to interact
# with Twitter from here on out:
api = tweepy.API(auth)
def remove_stopwords(text) -> str:
    """ Remove stopwords from text """
    filtered_words = [word for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

def expand_hashtag(tag: str):
    """ Convert #HashTag to separated words.
    '#ActOnClimate' => 'Act On Climate'
    '#climate' => 'climate' """
    res = re.findall('[A-Z]+[^A-Z]*', tag)
    return ' '.join(res) if len(res) > 0 else tag[1:]

def expand_hashtags(s: str):
    """ Convert string with hashtags.
    '#ActOnClimate now' => 'Act On Climate now' """
    res = re.findall(r'#\w+', s) 
    s_out = s
    for tag in re.findall(r'#\w+', s):
        s_out = s_out.replace(tag, expand_hashtag(tag))
    return s_out
def remove_last_hashtags(s: str):
    """ Remove all hashtags at the end of the text except #url """
    # "Change in #mind AP #News #Environment" => "Change in #mind AP"
    tokens = TweetTokenizer().tokenize(s)
    # If the URL was added, keep it
    url = "#url" if "#url" in tokens else None
    # Remove hashtags
    while len(tokens) > 0 and tokens[-1].startswith("#"):
        tokens = tokens[:-1]
    # Restore 'url' if it was added
    if url is not None:
        tokens.append(url)
    return ' '.join(tokens) 

def text_clean(s_text: str) -> str:
    """ Text clean """
    try:
        output = re.sub(r"https?://\S+", "#url", s_text)  # Replace hyperlinks with '#url'
        output = re.sub(r'@\w+', '', output)  # Remove mentioned user names @... 
        output = remove_last_hashtags(output)  # Remove hashtags from the end of a string
        output = expand_hashtags(output)  # Expand hashtags to words
        output = re.sub("[^a-zA-Z]+", " ", output) # Filter
        output = re.sub(r"\s+", " ", output)  # Remove multiple spaces
        output = remove_stopwords(output)  # Remove stopwords
        return output.lower().strip()
    except:
        return ""
def partial_clean(s_text: str) -> str:
    """ Convert tweet to a plain text sentence """
    output = re.sub(r"https?://\S+", "#url", s_text)  # Replace hyperlinks with '#url'
    output = re.sub(r'@\w+', '', output)  # Remove mentioned user names @... 
    output = remove_last_hashtags(output)  # Remove hashtags from the end of a string
    output = expand_hashtags(output)  # Expand hashtags to words
    output = re.sub(r"\s+", " ", output)  # Remove multiple spaces
    return output
def get_hashtags(text):
    return ','.join([word for word in text.split() if word[0] == '#'])
#Get the database credentials in order to connect
def getDatabaseConnection():
        return mySQL.connect(host=config['database']['host'], user=config['database']['user'], passwd=config['database']['passwd'], db=config['database']['db'], charset=config['database']['charset'], auth_plugin='mysql_native_password')
def handling_database():
    mydb = getDatabaseConnection()
    cursor = mydb.cursor()
    #word_vectors = gapi.load('word2vec-google-news-300')
    hashtags = ["agritech", "artificial intelligence", "biotech", "blockchain", 
                "cybersecurity", "drones", "industry-4.0", "internet-of-things", "robotics", 
                "virtual reality", "augmented reality","fintech", "medtech", "logistics", "micromobility", 
                "climate_tech", "greentech", "sustainability"]
    #if os.path.exists('file_twitter.json'):
        #os.remove('file_twitter.json')
    #with open('file_twitter.json', 'a') as fp:
        #json.dump("data:",fp, indent=4)
    with open('5000-words.txt') as f:
        lines = ''.join(f.readlines()).rstrip('\n').split()
    ids=[]
    cntr = -1
    stmt = "SHOW TABLES LIKE 'tweets'"
    cursor.execute(stmt)
    existment = cursor.fetchone()
    if existment:
        #pass
        #"""
        query = "UPDATE tweets SET is_processed = (%s)"#WHERE is_processed = %s
        values = (True)#, False
        cursor.execute(query, (values,))
        mydb.commit()
                
        cursor.execute("SELECT COUNT(*) FROM tweets")
        # Get the result of the query
        result = cursor.fetchone()   
        # The result is a tuple with one element, which contains the count
        row_count = result[0]

        cursor.execute("SELECT * FROM tweets")
        # Get the result of the query
        result_df = cursor.fetchall()   
        # The result is a tuple with one element, which contains the count
        print(result_df)
        print(type(result_df))
        cnt=0
    # UNIQUE         
    else:
        cursor.execute("CREATE TABLE tweets (id SERIAL PRIMARY KEY,tweet_id VARCHAR(255) NOT NULL,user_followers_count INT,full_text VARCHAR(255),hashtags VARCHAR(255), user_friends_count INT, retweets_count INT, favourites_count INT, tech_category VARCHAR(50),is_processed BOOLEAN)")
        row_count=0
    json_data=[]
    for hashtag in hashtags:
            qr = hashtag
            try:
                tweets = client.search_recent_tweets(query=qr,
                                        
                                        tweet_fields = ["created_at", "text", "source","entities","public_metrics"],
                                        user_fields = ["name", "username", "location", "verified", "description","public_metrics"],
                                        max_results = 50,
                                        expansions= ['attachments.poll_ids', 'attachments.media_keys', 'author_id', 'edit_history_tweet_ids', 
                                        'entities.mentions.username', 'geo.place_id', 'in_reply_to_user_id', 
                                        'referenced_tweets.id', 'referenced_tweets.id.author_id']#'entities.mentions.username'#'author_id'
                
                                    )

                """   
                ##print(tweets.data)
                print('--------------------------')
                print(hashtag)
                print(tweets.data)
                print('--------------------------')
                """
                cnt=0  
                for tweet in tweets.data:
                    cnt+=1
                    #try:
                    full_text = text_clean(tweet.text)
                    full_text = partial_clean(full_text)
                    query = "INSERT INTO tweets (id,tweet_id,user_followers_count,full_text,hashtags,user_friends_count,retweets_count,favourites_count,tech_category,is_processed)"
                    query += " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    values = (int(row_count+1), tweet.id,int(list(tweets.includes["users"][cnt].public_metrics.values())[0]),full_text,','.join([word[1:] for word in tweet.text.split() if word[0] == '#']),list(tweets.includes["users"][cnt].public_metrics.values())[1],list(tweet.public_metrics.values())[0],list(tweet.public_metrics.values())[2],hashtag,False)
                    cursor.execute(query, values)
                    mydb.commit()
                    # Get the result of the query
                    cursor.execute("SELECT COUNT(*) FROM tweets")
                    
                    result = cursor.fetchone()     
                    # The result is a tuple with one element, which contains the count
                    row_count+=1
                    cnt+=1
                    
                    #print(cnt)
            except Exception as e:
                    pass
            cursor.execute('''SELECT * FROM tweets''')
            rv = cursor.fetchall()
            row_headers=[x[0] for x in cursor.description]
            
            for un_result in rv:
                json_data.append(dict(zip(row_headers,un_result)))
    jn= json.dumps(json_data)
    with open("tweets.json", "w") as outfile:
        outfile.write(jn)

#handling_database()
#"""
schedule.every().day.at("00:00").do(handling_database)

while True:
 
    # Checks whether a scheduled task 
    # is pending to run or not
    schedule.run_pending()
    time.sleep(1000)
#"""
