from selenium import webdriver
import pandas as pd
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from urllib import request
import re
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
import mysql.connector as mySQL
import configparser
import requests
from bs4 import BeautifulSoup
#from flask import Flask, request
from urllib.request import Request, urlopen
import re
from datetime import date
import time
import json
from os import path
import os

today = date.today()
config = configparser.ConfigParser()
config.read('config.ini')




def get_articles():
    
    query = config['query']['new_query']
    print(query)
    if os.path.isdir("outputs"):
        
        print(3)
        if os.path.isdir(os.path.join(os.getcwd(), "outputs","research_output")):
            pass
            
        else:
            path = os.path.join(os.getcwd(), "outputs","research_output")
        
            
            os.mkdir(path)

    else:
        # Directory
        directory = "outputs"
        
        parent_dir = os.getcwd()
        
        # Path
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        path1 = os.path.join(parent_dir, directory,"research_output")
        os.mkdir(path1)
        
    
    if os.path.isfile(os.path.join(os.getcwd(), "outputs","research_output",'{}_1.json'.format(query))):
        file_json = open (os.path.join(os.getcwd(), "outputs","research_output",'{}_{}.json'.format(query,str(len(os.listdir(os.path.join(os.getcwd(), "outputs","research_output")))+1))),'w')
    else:
        file_json = open (os.path.join(os.getcwd(), "outputs","research_output",'{}_1.json'.format(query)),'w')
    
    start = time.time()
    
    listObj = ieee_articles(query)+articles_wiley(query)
    json_object = json.dumps(listObj)
    file_json.write(json_object)
    
    
    
def ieee_articles(query):
    

    ieee_json={}
    
 
    filename = os.getcwd()+'ieee.json'
    listObj = []
    
    
    
    href_list1=leg_links('sortType=newest',query)
    
    today = date.today()
    cur_year = str(today.year)
    pre_year = str(int(today.year)-1)
    
    href_list2=leg_links('ranges={}_{}_Year'.format(pre_year,cur_year),query)
    href_init = href_list1 + href_list2
    
    href_list = []

    [href_list.append(item) for item in href_init if item not in href_list]

    
    for item in href_list:
        init_json = {}
        try:
            response = get_source(item)
            soup = BeautifulSoup(response.content, 'html.parser')
            texts = soup.findAll(text=True)
            
            ft= ["abstract","Abstract"]
            fi = ["formulaStrippedArticleTitle","displayDocTitle"]
            for i in range(0,len((str(texts).split(':')))):  
                for k in fi:
                    if((str(texts).split(':'))[i].find(k)>0):
                        title = (str(texts).split(':'))[i+1].split(',')[0]
                        y={"Title":title}
                        tit = "".join([str(item) for item in (str(texts).split(':'))[i+1:i+3] ])#if len(item.split())>1
                        
                    break
            for i in range(0,len((str(texts).split(':')))):
                for j in ft:
                    if((str(texts).split(':'))[i].find(j)>0):
                        if len((str(texts).split(':'))[i+1].split()) > 10:
                            if len(title) != 0:
                                
                                res = "".join([str(item) for item in (((((str(texts).split(':'))[i+1])).split(',')[:-1]))])
                                
                                x = {"Abstract":res}
                               
                                                    
                            break
                                        
            listObj.append({"Query":str(query),"Title":str(title),"Text":str(res),"Source":"IEEE","Date":str(today)})
                                                          
                                
        except requests.exceptions.SSLError as e:
                            print(e)
   
 
    
    
    return listObj
    
def leg_links(string,query):
    

    chrome_options = Options()
    
    driver = webdriver.Chrome()
    
    driver.get("https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText={}&{}".format(query,string))
    time.sleep(10)
    elems = driver.find_elements(By.TAG_NAME, 'a')
    href_list = []
    title_list = []
    for elem in elems:
            
            if (elem.get_attribute('href') is not None) and('document'in (elem.get_attribute('href').split('/'))):#and((elem.get_attribute('href').split('/'))[-2]=='document')
                
                if (elem.get_attribute('href').split('#')[-1]) != 'citations' and (elem.get_attribute('href').split('#')[-1]) != 'anchor-patent-citations':
                    href_list.append(elem.get_attribute('href'))
                    href_list = list(set(href_list)) 
    return href_list


            
#
def get_source(url):
    """Return the source code for the provided URL. 

    Args: 
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html. 
    """

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)
def articles_wiley(query):
    
    listObj = []
    cur_year = str(today.year)
    pre_year = str(int(today.year)-1)
    
    chrome_options = Options()
    chrome_options.add_argument("-add_argument-window-size=1920,1080")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://onlinelibrary.wiley.com/action/doSearch?AllField="+ query+"&content=articlesChapters&target=default&AfterYear={}&BeforeYear={}".format(pre_year,cur_year))
    time.sleep(5)
    elems = driver.find_elements(By.TAG_NAME, 'a')
    elems3 = driver.find_elements(By.XPATH,"//div[@class='accordion']")#[@title='show Abstract']
    cnt=-1
    link_list = []
    wiley_json={}
    for elem in elems3:
            e="https://onlinelibrary.wiley.com/"+elem.get_attribute('innerHTML').split('<')[-2].split('?')[1].split()[0].split(';')[0].split('&')[0].replace("=", "/").replace("%2F", "/")
            link_list.append(e)
    for item in elems:
        if item.get_attribute("class") == 'publication_title visitable':
            cnt += 1
            chrome_options = Options()
            chrome_options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(link_list[cnt])
            time.sleep(5)
            try:
                elems3 = driver.find_element(By.XPATH,"//div[@class='article-section__content de main']")#[@title='show Abstract'] class="article-section__content de main"
                listObj.append({"Query":str(query),"Title":str(item.text),"Text":str(elems3.text),"Source":"Wiley","Date":str(today)})
                
            except NoSuchElementException:
                try:
                    elems3 = driver.find_element(By.XPATH,"//div[@class='article-section__content en main']")#[@title='show Abstract'] class="article-section__content de main"
                    listObj.append({"Query":str(query),"Title":str(item.text),"Text":str(elems3.text),"Source":"Wiley","Date":str(today)})
                    
                except NoSuchElementException:
                    pass
                
            time.sleep(5)
    return listObj

def getDatabaseConnection():
        return mySQL.connect(host=config['database']['host'], user=config['database']['user'], passwd=config['database']['passwd'], db=config['database']['db'], charset=config['database']['charset'], auth_plugin='mysql_native_password')

def get_source(url):
    """Return the source code for the provided URL. 

    Args: 
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html. 
    """

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)


if __name__ == '__main__':
    get_articles()


                       
 




