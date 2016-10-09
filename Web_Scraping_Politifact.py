# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 23:53:03 2016

@author: mohit
"""
#import the two libraries we will be using in this script
import urllib2,sys,re,time
from bs4 import BeautifulSoup
import pandas as pd

#make a new browser, this will download pages from the web for us. This is done by calling the 
#build_opener() method from the urllib2 library
browser=urllib2.build_opener()

#desguise the browser, so that websites think it is an actual browser running on a computer
#browser.addheaders=[('User-agent', 'Mozilla/5.0')]
browser.addheaders=[('User-agent', 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_1 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9B179 Safari/7534.48.3')]

#number of pages you want to retrieve
pagesToGet=198

#for every number in the range from 1 to pageNum+1

upperframe=[]  
for page in range(1,pagesToGet+1):
    print 'processing page :', page
      
    url = 'http://www.politifact.com/truth-o-meter/statements/?page='+str(page)
    #an exception might be thrown, so the code should be in a try-except block
    try:
        #use the browser to get the url. This is suspicious command that might blow up.
        response=browser.open(url)# this might throw an exception if something goes wrong.
        # response is equivalent to an enter key in chrome
    except Exception as e: # this describes what to do if an exception is thrown
        error_type, error_obj, error_info = sys.exc_info()# get the exception information
        print 'ERROR FOR LINK:',url #print the link that cause the problem
        print error_type, 'Line:', error_info.tb_lineno #print error info and line that threw the exception
        continue#ignore this page. Abandon this and go back.
    
    #read the response in html format. This is essentially a long piece of text
    myHTML=response.read()

    time.sleep(2)   
    tree=BeautifulSoup(myHTML)
    
    frame=[]
    
    politicsChunks = tree.findAll('div', {'class':'scoretable__item'}) 
    for PC in politicsChunks:
        
        #follow tree structure to pick the relevant tag
        temp= PC.contents[1].contents[1].contents[1].contents[0]
        
        #split based on regular expression
        validity=re.findall('<img alt=\"(.*?)" src=\"http',str(temp))[0]       
        
        sourcechunk=PC.find('div',{'class':'statement__source'})
        source=sourcechunk.text
        source = source.encode('ascii','ignore')   
                   
        statementchunk=PC.find('p',{'class':'statement__text'})
        statement = statementchunk.text
        statement =statement.encode('ascii','ignore')

        frame.append([source,statement.strip(),validity]) #add each record to page chunk frame
    upperframe.extend(frame)# add each page record set to master set
dataset=pd.DataFrame(upperframe) # convert into dataframe

# Convert dataframe to csv

dataset.to_csv('C:\Users\mohit\Desktop\Spring BIA\Web Analytics\Politifact\Politifact.csv')
      