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
#import cv2
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
#import mysql.connector as mySQL
#import mysql.connector.errors
#####AUTO TO SCRIPT EINAI GIA THN DHMIOURGIA CLUSTER ME VASI DEDOMENWN. EINAI MEXRI STIGMIS EINAI H TELIKH EKDOSH 21/11/2023

stop = set(stopwords.words("english"))

import spacy
import nltk
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
config = configparser.ConfigParser()
config.read('configuration.ini')



def text_filter(s_data: str) -> str:
    """ Remove extra characters from text """
    return s_data.replace("&amp;", "and").replace(";", " ").replace(",", " ") \
                 .replace('"', " ").replace("\n", " ").replace("  ", " ")

def get_hashtags(tweet) -> str:
    """ Parse retweeted data """
    hash_tags = ""
    if 'hashtags' in tweet.entities:
        hash_tags = ','.join(map(lambda x: x["text"], tweet.entities['hashtags']))
    return hash_tags

def get_csv_header() -> str:
    """ CSV header """
    return "id;created_at;user_name;user_location;user_followers_count;user_friends_count;retweets_count;favorites_count;retweet_orig_id;retweet_orig_user;hash_tags;full_text;".rstrip()

def tweet_to_csv(tweet):
    """ Convert a tweet data to the CSV string """
    if not hasattr(tweet, 'retweeted_status'):
        full_text = text_filter(tweet.full_text)
        hasgtags = get_hashtags(tweet)
        retweet_orig_id = ""
        retweet_orig_user = ""
        favs, retweets = tweet.favorite_count, tweet.retweet_count
    else:
        retweet = tweet.retweeted_status
        retweet_orig_id = retweet.id
        retweet_orig_user = retweet.user.screen_name
        full_text = text_filter(retweet.full_text)
        hasgtags = get_hashtags(retweet)
        favs, retweets = retweet.favorite_count, retweet.retweet_count
    s_out = f"{tweet.id};{tweet.created_at};{tweet.user.screen_name};{tweet.user.location};{tweet.user.followers_count};{tweet.user.friends_count};{retweets};{favs};{retweet_orig_id};{retweet_orig_user};{hasgtags};{full_text}"
    return s_out
# Execute a SQL query that counts the number of rows in the table



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

def lemmatize(sentence: str) -> str:
    """ Convert all words in sentence to lemmatized form """
    return " ".join([token.lemma_ for token in nlp(sentence)])

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

def text_len(s_text: str) -> int:
    """ Length of the text """
    return len(s_text)


def draw_cloud(column: pd.Series, stopwords=None):
    all_words = ' '.join([text for text in column]) 
    
    wordcloud = WordCloud(width=1600, height=1200, random_state=21, max_font_size=110, collocations=False, stopwords=stopwords).generate(all_words) 
    plt.figure(figsize=(16, 12)) 
    plt.imshow(wordcloud, interpolation="bilinear") 
    plt.axis('off')
    #plt.show()

def partial_clean(s_text: str) -> str:
    """ Convert tweet to a plain text sentence """
    output = re.sub(r"https?://\S+", "#url", s_text)  # Replace hyperlinks with '#url'
    output = re.sub(r'@\w+', '', output)  # Remove mentioned user names @... 
    output = remove_last_hashtags(output)  # Remove hashtags from the end of a string
    output = expand_hashtags(output)  # Expand hashtags to words
    output = re.sub(r"\s+", " ", output)  # Remove multiple spaces
    return output
def word2vec_vectorize(text: str):
    """ Convert text document to the embedding vector """
    word_vectors = gapi.load('word2vec-google-news-300')    
    vectors = []
    tokens = word_tokenize(text)
    for token in tokens:
        if token in word_vectors:
            vectors.append(word_vectors[token])
            
    return np.asarray(vectors).mean(axis=0) if len(vectors) > 0 else np.zeros(word_vectors.vector_size)
def make_clustered_dataframe(x: np.array, k: int) -> pd.DataFrame:
    """ Create a new dataframe with original docs and assigned clusters """
    ids = df_new["id"].values
    #user_names = df_new["user_name"].values
    docs = df_new["text_clean"].values
    tokenized_docs = df_new["text_clean"].map(text_to_tokens).values
    
    km = KMeans(n_clusters=k).fit(x)
    s_score = silhouette_score(x, km.labels_)
    print(f"K={k}: Silhouette coefficient {s_score:0.2f}, inertia:{km.inertia_}")
    
    # Create new DataFrame
    data_len = x.shape[0]
    df_clusters = pd.DataFrame({
        "id": ids[:data_len],
        #"user": user_names[:data_len],
        "text": docs[:data_len],
        "tokens": tokenized_docs[:data_len],
        "cluster": km.labels_,
    })
    return df_clusters


def text_to_tokens(text: str) -> List[str]:
    """ Generate tokens from the sentence """
    # "this is text" => ['this', 'is' 'text']
    tokens = word_tokenize(text)  # Get tokens from text
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens
def show_clusters_info(term:str,x: np.array, k: int, cdf: pd.DataFrame):
    """ Print clusters info and top clusters """
    labels = cdf["cluster"].values
    sample_silhouette_values = silhouette_samples(x, labels)
    total_vocab = Counter()
    s_all_total = ""
    # Get silhouette values per cluster    
    silhouette_values = []
    for i in range(k):
        cluster_values = sample_silhouette_values[labels == i]
        silhouette_values.append((i, 
                                  cluster_values.shape[0], 
                                  cluster_values.mean(), 
                                  cluster_values.min(), 
                                  cluster_values.max()))
    # Sort
    silhouette_values = sorted(silhouette_values, 
                               key=lambda tup: tup[2], 
                               reverse=True)
    
    # Show clusters, sorted by silhouette values
    for s in silhouette_values:
        print(f"Cluster {s[0]}: Size:{s[1]}, avg:{s[2]:.2f}, min:{s[3]:.2f}, max: {s[4]:.2f}")
    
    # Show top 7 clusters
    top_clusters = []
    cnt=0
    for cl in silhouette_values[:4]:#7
        cnt+=1
        df_c = cdf[cdf['cluster'] == cl[0]]   
        s_all = ""
        for tokens_list in df_c['tokens'].values:
            s_all += ' '.join([text for text in tokens_list]) + " "
            s_all_total += ' '.join([text for text in tokens_list]) + " "           
        wordcloud = draw_cloud_from_words(s_all, stopwords=["url"])
        wordcloud.to_file("wordcloud_{}.png".format(cnt))
        # Show most popular words
        for token in df_c["tokens"].values:
            if token != 'url':
                #vocab.update(token)
                total_vocab.update(token)
    print(s_all_total)
    with open("{}.json".format(term), "w") as outfile:
        outfile.write(s_all_total)
    return s_all_total, total_vocab

def draw_cloud_from_words(all_words: str, stopwords=None):
    """ Show the word cloud from the list of words """
    wordcloud = WordCloud(width=1600, height=1200, random_state=21, max_font_size=110, collocations=False, stopwords=stopwords).generate(all_words) 
    return wordcloud




def graw_elbow_graph(x: np.array, k1: int, k2: int, k3: int):
    k_values, inertia_values = [], []
    for k in range(k1, k2, k3):
        print("Processing:", k)
        km = KMeans(n_clusters=k).fit(x)
        k_values.append(k)
        inertia_values.append(km.inertia_)

    plt.figure(figsize=(12,4))
    plt.plot(k_values, inertia_values, 'o')
    plt.title('Inertia for each K')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.show()


def draw_clusters_tsne(docs: List, cdf: pd.DataFrame):
    """ Draw clusters using TSNE """
    cluster_labels = cdf["cluster"].values
    cluster_names = [str(c) for c in cluster_labels]
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300, 
                init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(vectorized_docs_trans)

    # Plot output
    x, y = tsne_results[:, 0], tsne_results[:, 1]
    source = ColumnDataSource(dict(x=x, 
                                   y=y, 
                                   labels=cluster_labels,
                                   colors=cluster_names))
    palette = (RdYlBu11 + BrBG11 + Viridis11 + Plasma11 + Cividis11 + RdGy11)[:len(cluster_names)]

    p = figure(width=1600, height=900, title="")
    p.scatter("x", "y",
              source=source, fill_alpha=0.8, size=4,
              legend_group='labels',
              color=factor_cmap('colors', palette, cluster_names)
              )
    show(p)
    
def get_all_data():
    f = open ('nested_data.json', "r")
    df = pd.DataFrame(json.loads(f.read()))
    print(df)
    return df

def main(dataframe):
    
    hashtags = ["agritech", "artificial intelligence", "biotech", "blockchain", 
                "cybersecurity", "drones", "industry-4.0", "internet-of-things", "robotics", 
                "virtual reality", "augmented reality","fintech", "medtech", "logistics", "micromobility", 
                "climate_tech", "greentech", "sustainability"]
    if os.path.exists('file_twitter.json'):
        os.remove('file_twitter.json')
    with open('file_twitter.json', 'a') as fp:
        json.dump("data:",fp, indent=4)
    with open('5000-words.txt') as f:
        lines = ''.join(f.readlines()).rstrip('\n').split()
    ids_final =[]
    cntr = -1
    df = dataframe.copy()
    df['is_processed']=0
    #print(df['tech_category'].unique()[0])
    #hstg=df['tech_category'].unique()[0]
    for hstg in df['tech_category'].unique():
        print(hstg)
    
        try:
            new_jsn={}
            cntr+=1
            df_new = df[(df['tech_category']==hstg) & (df['is_processed']==0)]
            print(df_new)

            if len(df_new)>=3:
                df_new['text_clean'] = df_new.loc[:,'full_text'].map(text_clean)
                df_new['text_len'] = df_new['text_clean'].map(text_len)
                df_new = df_new[df_new['text_len'] > 32]
                print('================================')
                print(df_new)
                
                docs = df_new["text_clean"].values
                docs = df_new['full_text'].map(partial_clean).values
                print(docs)
                
                
                # Make clustered dataframe
                #"""
                k = 3#0
                if len(df_new)>=3:
                    print(df_new.columns)
                    
                    word_vectors = gapi.load('word2vec-google-news-300')
                    vectorized_docs_w2v = np.asarray(list(map(word2vec_vectorize, docs)))
                    model = SentenceTransformer('all-MiniLM-L6-v2')

                    ids = df_new["id"].values
                    #user_names = df_new["user_name"].values
                    docs = df_new["text_clean"].values
                    tokenized_docs = df_new["text_clean"].map(text_to_tokens).values
                    
                    km = KMeans(n_clusters=k).fit(vectorized_docs_w2v)
                    s_score = silhouette_score(vectorized_docs_w2v, km.labels_)
                    print(f"K={k}: Silhouette coefficient {s_score:0.2f}, inertia:{km.inertia_}")
                    
                    # Create new DataFrame
                    data_len = vectorized_docs_w2v.shape[0]
                    df_clusters = pd.DataFrame({
                        "id": ids[:data_len],
                        #"user": user_names[:data_len],
                        "text": docs[:data_len],
                        "tokens": tokenized_docs[:data_len],
                        "cluster": km.labels_,
                    })

                    #df_clusters_w2v = df_clusters #make_clustered_dataframe(vectorized_docs_w2v, k)

                    #show_clusters_info(hstg,vectorized_docs_w2v, k, df_clusters_w2v)
                    #wds,countern = show_clusters_info(hstg,vectorized_docs_w2v, k, df_clusters_w2v)#[0]
                    wds,countern = show_clusters_info(hstg,vectorized_docs_w2v, k, df_clusters)#[0]
                    print(wds)

                    word_set = [word for word, occurrences in countern.items() if occurrences >= 2]
                    word_pack = [wd for wd in wds.split() if wd in word_set]
                    word_pack = [wd for wd in word_pack if wd not in lines]
                    word_pack = [wd for wd in word_pack if wd !='url'and wd !='rt']
                    word_set=''
                    for item in word_pack:
                        word_set+=" {}".format(item)
                    new_jsn["id"]=str(cntr)
                    new_jsn["category"]=hstg
                    new_jsn["text"]=word_set
                    ids_final.append(new_jsn)

                
            
          
        except Exception as e:
            print('---------------------')
            print(e)
            print('---------------------')
            #pass
       
    print(ids_final)
    with open('file_twitter.json', 'a') as fp:
        json.dump(ids_final,fp, indent=4)
    return "data:{}".format(str(ids_final))
    
    
if __name__ == '__main__':
    main(get_all_data())
    #get_all_data()
    


