import pandas as pd
import plotly.express as px
import datetime
import os
import json
import configparser
import csv 
import json 
import tweepy
import mysql.connector as mySQL
import mysql.connector.errors
import gensim.downloader as gapi

config = configparser.ConfigParser()
config.read('configuration.ini')


def get_all_data():
    f = open ('nested_data.json', "r")
    df = pd.DataFrame(json.loads(f.read()))
    return df
def main(dataframe):
    hashtags = ["agritech","artificial intelligence", "biotech", "blockchain", 
                    "cybersecurity", "drones", "industry-4.0", "internet-of-things", "robotics", 
                    "virtual reality","fintech", "medtech", "logistics", "micromobility"]#"agritech", , "biotech", "blockchain", "cybersecurity", "drones", "industry-4.0", "internet-of-things", "robotics", "vr-ar", "fintech", "medtech", "logistics", "micromobility", "climate_tech", "greentech", "sustainability"]
        
    jsn_list = []   
    for hashtag in hashtags:
        df = dataframe.copy()
        ids=[]
        cntr = -1
        df['value'] = (((df['user_followers_count']) + (df['user_friends_count'])/10)/2)+10*(df['retweets_count'])+100*(df['favourites_count'])
        df['value'] = ((df['value']-df['value'].mean())/df['value'].std())
        df11 = df[df["hashtags"].isin((df['hashtags'].value_counts().nlargest(7)).index.values.tolist())].reset_index(drop=True)[["user_followers_count","retweets_count", "hashtags", "value"]]
        df13 = df11.groupby(["hashtags"]).size().reset_index().rename(columns={0:'count'})
        df_final=pd.DataFrame()
        df_final_1=pd.DataFrame()
        for j in range(0,len(df13)):
            dfn = df11[(df11['hashtags'] == df13.iloc[j]['hashtags'])]
            df_final_1 = df_final_1.append({'user_followers_count':round((dfn[dfn['hashtags']==df13.iloc[j]['hashtags']]['user_followers_count'].sum())/(df13.iloc[j]['count']),0),'retweets_count':int((dfn[dfn['hashtags']==df13.iloc[j]['hashtags']]['retweets_count'].sum())/(df13.iloc[j]['count'])),'hashtags' : df13.iloc[j]['hashtags'], 'value' : int(((dfn['value'].sum())/(len(dfn)))*100)}, ignore_index = True)
            
        for k in range(0, len(df_final_1)):
            jsn={}
            jsn["x"]=df_final_1.iloc[k]['user_followers_count']
            jsn["y"]=df_final_1.iloc[k]['retweets_count']
            jsn["z"]=abs(df_final_1.iloc[k]['value'])
            jsn["name"]=df_final_1.iloc[k]['hashtags']
            jsn["category"]=hashtag
            jsn_list.append(jsn)    
    
    
    with open(os.getcwd()+"/new_results/bubbles_new.json", "w") as outfile:#.format(term)
        json.dump("data:{}".format(str(jsn_list)), outfile)
        
    outfile.close()
    return "data:{}".format(str(jsn_list))

if __name__ == '__main__':
    main(get_all_data())