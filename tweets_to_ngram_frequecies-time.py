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
from itertools import cycle
from pathlib import Path


# PART 0 - Read the data and get a list of tweets that you want to examine
def getTweets(day, path):
    # read the data as a dataframe
#    df=pd.read_csv('daily_results/daily_csv_files/'+day,sep='\t',quoting=3)
    df=pd.read_csv(path+day, sep='\n', quoting=3, header=None, names=["Text"])

    #get the tweets
    tweets=df["Text"].tolist()
    return tweets


# PART 1 - Read each tweet and identify phrases separated by punctuation
def separatePhrases(tweets):    
    #This is a list of 'stop' characters - might need updating as we find more things that mark the end of a phrase
    punctuation=['.',',',';',':','"','(',')','[',']','{','}','¿','?','-','!','\\','/','-']
    
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
    return phrases

        
# PART 2 - Take each phrase and count the 1, 2, 3 ,4 and 5 grams    
def countNgrams(phrases):
    # create a dictionary of dictionaries so that grams[n] will be the dictionary for ngrams
    grams={1:{},2:{},3:{},4:{},5:{},6:{}}
    
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
    return grams



# PART 3 - Write the results to file
def writeResults(grams):    
    for n in grams:
        frequencies=[]
        for ngram in grams[n]:
            frequencies.append([ngram,grams[n][ngram]])
            
        # sort the list so that highest frequencies are on top
        frequencies=sorted(frequencies,key=lambda item: item[1],reverse=True)
        
        output_file = open('daily_results/'+str(n)+'grams/'+str(day[0:8]),'w')

        for f in frequencies:
            ngram=f[0]
            frequency=f[1]
            str1=ngram+'\t'+str(frequency)
            str1=str1+'\n'
            output_file.write(str1)
        output_file.close()

        
# PART 3 - Write the clusterized results to file
def writeResultsClusters(grams, name, path):
    for n in grams:
        frequencies=[]
        for ngram in grams[n]:
            frequencies.append([ngram,grams[n][ngram]])

            # sort the list so that highest frequencies are on top
        frequencies=sorted(frequencies,key=lambda item: item[1],reverse=True)

        p = path+str(n)+'grams/'
        if not os.path.exists(p):
            os.makedirs(p)
        output_file = open(p+name,'w')
        print(output_file)
        print("frec len: " + str(len(frequencies)))
        for f in frequencies:
            ngram=f[0]
            frequency=f[1]
            str1=ngram+'\t'+str(frequency)+'\n'
            output_file.write(str1)
        output_file.close()        

        
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ngrams_per_days(days, pathCSV, pathOutput, Ndays):
    clusters = list(chunks(range(len(days)), Ndays))
    for i in clusters:
        phrases = []
        name = ""
        for day in i:
            d = days[day]
            print(d)
            name = d[0:8]
            tweets = getTweets(d, pathCSV)
            s = separatePhrases(tweets)
            phrases.extend(s)
        grams = countNgrams(phrases)
        writeResultsClusters(grams, name, pathOutput)

        
def ngrams_timelapse(timelapse):
        j = list(journals.keys())
        for journal in j:
            start = journals[journal]['start_volume']
            end = journals[journal]['end_volume']
            time = journals[journal]['timelapse']
            volumes = [i for i in range(start, end+1) for _ in range(time)]
            chunk = list(zip(volumes, cycle(months)))
            distribution = list(chunks(chunk, timelapse))
            for l in distribution:
                phrases = []
                for pair in l:
                    volume = str(pair[0])
                    month = pair[1]
                    pathVolumes = "/home/fer/aps/pdf_texts/"+journal+"/"+volume+"/"+month+"/"
                    if Path(pathVolumes).exists():
                        entries = os.listdir(pathVolumes)
                        for day in entries:
                            tweets = getTweets(day, pathVolumes)
                            s = separatePhrases(tweets)
                            phrases.extend(s)
                grams = countNgrams(phrases)
                pathOutput = "results/" + journal + "/" + str(timelapse) + "months/" 
                name = str(l[0][0])+"_"+l[0][1]
                writeResultsClusters(grams, name, pathOutput)

                                                               
def getTweetsByHours(day, path, nhours):
    l_frame = []
    df=pd.read_csv(path+day,sep='\t',quoting=3)
    print(path+day)
    startTime = df["Time"][0]
    last_time = df["Time"][len(df.index)-1]
    
    # select dates you want to examine
    delta = nhours * 3600

    while(startTime <= last_time):
        endTime = startTime + delta

        # read the data as a dataframe
#        df=pd.read_csv(path+day,sep='\t',quoting=3)

        #select the data within these boundaries
        df_new=df[(df['Time']<endTime) & (df['Time']>=startTime)]
        l_frame.append(df_new["Text"].tolist())
        startTime = endTime
    return l_frame



def ngrams_per_hours(days, pathCSV, pathOutput, N):
    for day in days:
        tweets = getTweetsByHours(day, pathCSV, N)
        count = 0
        for tweet in tweets:
            d = day
            print(d)
            name = d[0:8] + "-" + str(count)
            s = separatePhrases(tweet)
            grams = countNgrams(s)
            writeResultsClusters(grams, name, pathOutput+str(N)+'hours/')
            count += 1
        

path = "pdf_texts/"
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
interval = [1,3,6,9,12,24,36,48,60]

journals = {'pra':{'start_year':'1970', 'start_volume':1, 'end_year':'2017', 'end_volume':96, 'timelapse':6},
            'prb':{'start_year':'1970', 'start_volume':1, 'end_year':'2017', 'end_volume':96, 'timelapse':6},
            'prc':{'start_year':'1970', 'start_volume':1, 'end_year':'2017', 'end_volume':96, 'timelapse':6},
            'prd':{'start_year':'1970', 'start_volume':1, 'end_year':'2017', 'end_volume':96, 'timelapse':6},
            'pre':{'start_year':'1993', 'start_volume':47, 'end_year':'2017', 'end_volume':96, 'timelapse':6},
            'prl':{'start_year':'1959', 'start_volume':2, 'end_year':'2017', 'end_volume':119, 'timelapse':6},
            'prx':{'start_year':'2012', 'start_volume':2, 'end_year':'2017', 'end_volume':7, 'timelapse':12},
            'rmp':{'start_year':'1930', 'start_volume':2, 'end_year':'2017', 'end_volume':89, 'timelapse':12}}


for v in interval:
    ngrams_timelapse(v)






