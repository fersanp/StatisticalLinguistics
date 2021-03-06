import glob
import sys
import pandas as pd
import os.path
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# PART 0 - Read the data and get a list of tweets that you want to examine
def getTweets(day, path):
        # read the data as a dataframe
        #    df=pd.read_csv('daily_results/daily_csv_files/'+day,sep='\t',quoting=3)
    df=pd.read_csv(path+day, sep='\n', quoting=3, header=None, names=["Text"], engine='python')

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


def plot_success(percent):
    print(percent)
    fig, ax = plt.subplots()
    for k, v in sorted(percent.items()):
        sorted_x = sorted(v.items(), key=lambda kv: kv[0])
        res = [[ i for i, j in sorted_x ], [ j for i, j in sorted_x ]]
        plt.plot(res[1], '-o', label=str(k)+" words")

    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,100))
    plt.xticks(range(len(res[1])), ticks, rotation=-30)
    plt.legend()
    plt.title("Correctly Classified Articles")
    plt.xlabel("Journals")
    plt.ylabel("Percentages")
    plt.tight_layout()
    plt.savefig(outpath+"rate.png")
    



outpath = "predictionsResults/"
search_file = "importantWordsPerJournal/*pr[a-e]*1grams*"
path = "../../aps/pdf_texts/"
interval = [1]
journals = ["pra", "prb", "prc", "prd", "pre"]
ticks = ["Phys. Rev. A", "Phys. Rev. B", "Phys. Rev. C", "Phys. Rev. D", "Phys. Rev. E"]
volumes = ["95", "96"] # year 2017
Nwords = [10, 100, 1000, 2000]
#Nwords = [10]
Narchives = 500
#Narchives = 10
rates = {}


#r = {10:{'pra':76.4,'prb':18.2,'prc':44.4,'prd':49.2,'pre':49.2},
#     100:{'pra':59.8,'prb':72.8,'prc':72.0,'prd':79.4,'pre':87.4},
#     1000:{'pra':80.4,'prb':69.2,'prc':85.4,'prd':85.6,'pre':86.0},
#     2000:{'pra':87.4,'prb':70.2,'prc':86.2,'prd':90.6,'pre':90.6}}

r = {10: {'pra': 377/Narchives*100, 'prb': 101/Narchives*100, 'prc': 223/Narchives*100, 'prd': 239/Narchives*100, 'pre': 247/Narchives*100}, 
     100: {'pra': 287/Narchives*100, 'prb': 362/Narchives*100, 'prc': 361/Narchives*100, 'prd': 415/Narchives*100, 'pre': 441/Narchives*100}, 
     1000: {'pra': 424/Narchives*100, 'prb': 344/Narchives*100, 'prc': 420/Narchives*100, 'prd': 432/Narchives*100, 'pre': 441/Narchives*100}, 
     2000: {'pra': 438/Narchives*100, 'prb': 363/Narchives*100, 'prc': 415/Narchives*100, 'prd': 443/Narchives*100, 'pre': 449/Narchives*100}}
plot_success(r)
sys.exit(1)


for N in Nwords:
    percentages = {"pra":0, "prb":0, "prc":0, "prd":0, "pre":0}
    statistics = {"pra":0, "prb":0, "prc":0, "prd":0, "pre":0}
    false_positive = np.zeros(shape=(len(journals), len(journals)), dtype = int)
    for journal in journals:
        print(journal)
        testArchives = []
        flat_filenames = []
        for volume in volumes:
            filenames = os.walk(path+journal+"/"+volume+"/")
            fil = list(filenames)
            for tupla in list(fil):
                for item in tupla[2]:
                    testArchives.append((item,tupla[0]+"/"))
        random.shuffle(testArchives)
        testArchives = random.sample(testArchives, Narchives)
        

        for f in testArchives:
            print(f)
            tweets = getTweets(f[0], f[1])
            sep = separatePhrases(tweets)
            flat_list = [item for sublist in sep for item in sublist]
            s = set(flat_list)
            search_words = list(getImportantWords(s, "functional_words"))

            total = {"pra":0, "prb":0, "prc":0, "prd":0, "pre":0}
            
            res = [f for f in glob.glob(search_file)]
            for file in res:
                jour = file.split("_")[1]
                text = open(file, 'r')
                lines = text.readlines()[:N]
                for line in lines:
                    ln = line.split()
                    important_word = ln[0]
                    if important_word in search_words:
                        total.update({jour: total[jour]+1})

            print(total)
            print("-------------")
            key_max = max(total.items(), key=lambda x: x[1])
            if key_max[0] == journal:
                percentages[journal] += 1
            else:
                error = journals.index(key_max[0])
                idx = journals.index(journal)
                false_positive[idx][error] += 1

    print("##################")
    print("Success: ", percentages)
    print("False Positives")
    print(false_positive)

    
    for journal in journals: 
        statistics[journal] = round((percentages[journal]*100)/Narchives, 2)
    print("Statistics ", statistics)

    rates[N] = statistics


#--------------
## Heat Map of failures
    print("$$$$$$$$$$$$$$$$$$")
    fig, ax = plt.subplots()
    im = ax.matshow(false_positive, vmin=0, vmax=Narchives)
#im = ax.imshow(false_positive)
    ax.figure.colorbar(im)

# We want to show all ticks...
    ax.set_xticks(np.arange(len(journals)))
    ax.set_yticks(np.arange(len(journals)))
# ... and label them with the respective list entries
    ax.set_xticklabels(ticks, rotation=-30)
    ax.set_yticklabels(ticks)

# Loop over data dimensions and create text annotations.
    for i in range(len(journals)):
        for j in range(len(journals)):
            text = ax.text(j, i, false_positive[i, j],
                           ha="center", va="center", color="w")



#    ax.suptitle("False Positives\nTop "+str(N)+" words\n"+str(Narchives)+" Archives Tested", size=16)
    ax.set_title("False Positives Top "+str(N)+" words. "+str(Narchives)+" Archives Tested.", y=1.2)

    plt.xlabel("Errors")
    plt.ylabel("Journal")
    stri = "{"
    for key, value in statistics.items():
        stri += key+":"+str(value)+"%, "
    stri += "}"  
    print(stri)
    plt.text(0.10, 0.01 ,"Success Rate: "+stri, transform=plt.gcf().transFigure)
#    plt.text(0.25, 0.01 ,"Success Rate: "+str(percentages), transform=plt.gcf().transFigure)
    plt.tight_layout()

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.savefig(outpath+"predictions"+str(N)+".png")



plot_success(rates)
