# 27 April 2018 - Ewan Colman
# Code for reading twitter data from the excel sheet and ouputting ngram frequency lists.
#
# The file 'DataSample_IIMAS.xls' was first saved as a text file (.csv) with 
# utf-18 encoding and saved in the same directory as the script.
# An output directory Sample_results in the same folder as the script.
# 
# The output of this script is five text files named 1grams, 2grams, etc. that 
# contain ordered lists of ngrams and their corresponding frequencies 


# Modules required: 
import pandas as pd
import os
import datetime
import pytz
import year
from pathlib import Path


def frequencies(timelapse, journal, volume, month):
    grams={1:{},2:{},3:{},4:{},5:{},6:{}}
#for journal in os.listdir('pdf_texts/'):
    pathVolumes = "pdf_texts/"+journal+"/"+volume
    for m in month:
        if Path(pathVolumes+"/"+m+"/").exists():
            entries = os.listdir(pathVolumes+"/"+m+"/")
            for day in entries:
                print(m)
                print(day)
    # PART 0 - Read the data and get a list of tweets that you want to examine
    
    # read the data as a dataframe
    #df=pd.read_csv('DataSample_IIMAS.csv',encoding='utf-16')
                df=pd.read_csv(pathVolumes+"/"+m+"/"+day, sep="\n", quoting=3, header=None, names=["Text"])
#    print(df.head())

     #get the tweets
                tweets=df["Text"].tolist()
    
    
    # PART 1 - Read each tweet and identify phrases separated by punctuation
    
    #This is a list of 'stop' characters - might need updating as we find more things that mark the end of a phrase
                punctuation=['.',',',';',':','"','(',')','[',']','{','}','¿','?','-','!','\\','/']
    
    # we first create a list of phrases in the tweet
                phrases=[]
    
                for tweet in tweets:
                    tweet=str(tweet)
        # get the individual tokens
                    words=tweet.split()
        #print(tweet)
        #the phrase is a list of words 
                    phrase=[]
        #words=[]
                    new_phrase=False
                    while len(words)>0:
            #pop the first word on the list
                        word=words.pop(0)
           
            # if a new phrase has begun then save the old one    
                        if new_phrase==True:
                            phrases.append(phrase)
                            phrase=[]    
                        new_phrase=False
            
            #remove any punctuation characters from the beginning
                        while word[0] in punctuation and word not in punctuation:
                # remove the 0th character from the word 
                            word=word[1:len(word)]
                #start a new phrase
                            new_phrase=True
        
            # if a new phrase has begun then save the old one               
                        if new_phrase==True:
                            phrases.append(phrase)
                            phrase=[]
                        new_phrase=False
            
            #remove any punctuation characters from the end
                        while word[len(word)-1] in punctuation and word not in punctuation:
                #remove th last character from the word
                            word=word[0:len(word)-1]
                #start a new phrase
                            new_phrase=True
            
            # add word to phrase but not if it is a single puntuation character
                        if word not in punctuation:
                # change all characters to lower case
                            phrase.append(word.lower())
        
        # need to add the last phrase
                    phrases.append(phrase)
              
    # PART 2 - Take each phrase and count the 1, 2, 3 ,4 and 5 grams    
        
    # create a dictionary of dictionaries so that grams[n] will be the dictionary for ngrams
#                grams={1:{},2:{},3:{},4:{},5:{}}
    
                for phrase in phrases:
                    for n in grams:
            # here we create the ngrams (a phrase of length k will contain exactly k-n+1 ngrams)
                        for i in range(len(phrase)-n+1):
                # create the ngram one word at a time
                            ngram=''
                            for word in phrase[i:i+n]:
                    # insert "--" between each word of the ngram
                                ngram=ngram+word+'--'
                # remove the final "--"
                            ngram=ngram[0:len(ngram)-2]
                # update the frequencies 
                            if ngram in grams[n]:
                                grams[n][ngram]=grams[n][ngram]+1
                            else:
                                grams[n][ngram]=1
    
    
    # PART 3 - Write the results to file
    for n in grams:
        frequencies=[]
        for ngram in grams[n]:
            frequencies.append([ngram,grams[n][ngram]])
            
        # sort the list so that highest frequencies are on top
        frequencies=sorted(frequencies,key=lambda item: item[1],reverse=True)
        if frequencies != []:
            p = 'results/' + journal + "/" + timelapse + "/" + str(n)+'grams/'
            if not os.path.exists(p):
                os.makedirs(p)
#            day = day.replace(".txt", "")
            output_file = open(p+volume+"_"+month[0], 'w')
            
            for f in frequencies:
                ngram=f[0]
                frequency=f[1]
                str1=ngram+'\t'+str(frequency)
                str1=str1+'\n'
                output_file.write(str1)
            output_file.close()    


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


path = "pdf_texts/"
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
interval = [1,3,6,9,12,24,36,48,60]
for v in interval:
    for journal in os.listdir(path):
        for volume in os.listdir(path+"/"+journal):
            for i in chunks(months, v):
                frequencies(str(v)+"months", journal, volume, i)
