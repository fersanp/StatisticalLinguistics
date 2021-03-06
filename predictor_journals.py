import glob
import pandas as pd
import os



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


def getCommonWordsInAllJournals(ngram):
    path = "importantWordsPerJournal/"

    # get Ntop words in all journals
    common = set()
    for arch in glob.glob(path+'*'+ngram+'*.txt'):
    #        print(arch)
        words = open(arch, 'r').read().splitlines()
        #        words = words[:nTop]
        f2 = [w.split('\t', 1)[0] for w in words]

        if len(common) == 0:
            common.update(set(f2))
        else:
            common = common.intersection(set(f2))
    return common


def getImportantWords(results, infile):
    notImportant = open(infile, 'r').read().splitlines()
    notImportant = set(notImportant)
    
    N = 100
    english = open("english_words/1grams_2008_english_1000_words", "r").read().splitlines()
    for i in range(N):
        notImportant.add(english[i].split("\t")[0])
        
#    common = getCommonWordsInAllJournals("1grams")
#    notImportant.update(common)
        
    tmp = []
    for word in results:
        flag = True
        if len(word) > 2:
            word = word.lower()
            try:
                if word.encode('ascii').isalpha():
                    tmp.append(word)
            except:
                flag = False

    items = set(tmp)
    r = items.difference(notImportant)
    return r



search_file = "importantWordsPerJournal/*pr[a-e]*1grams*"
path = "pdf_texts/"
interval = [1]
months = ["January"]
journals = ["pra"]
volumes = ["80"]

pathCSV = "pdf_texts/pra/12/November/"
d = "PhysRevA.12.2031.txt"
tweets = getTweets(d, pathCSV)
sep = separatePhrases(tweets)
flat_list = [item for sublist in sep for item in sublist]
s = set(flat_list)
search_words = list(getImportantWords(s, "functional_words"))

                                
total = {"pra":0, "prb":0, "prc":0, "prd":0, "pre":0}
for search_word in search_words:
    print(search_word)
    results = []
    res = [f for f in glob.glob(search_file)]
    for file in res:
        found = False
        text = open(file, 'r')
        line = text.readline()
        j = 1
        while(not found and line):
            ls = line.split()
            word = ls[0]
            if word == search_word:
                found = True
                results.append((file, j))
            line = text.readline()
            j += 1
    print(results)

    min = results[0][1]
    min_journal = results[0][0]
    for i in results:
        if i[1] < min:
            min = i[1]
            min_journal = i[0]

    jour = min_journal.split("_")[1]
    total.update({jour: total[jour]+1})
#    print(search_word, min_journal)
#    print(total)
#    print()

print(total)
key_max = max(total.items(), key=lambda x: x[1])
print(key_max)


