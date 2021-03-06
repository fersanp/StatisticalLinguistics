import os
import operator
import glob

def merge_results(path, journal, timelapse, grams):
    results = {}
    path_complete = path+journal+"/"+timelapse+"/"+grams+"/"
    for file in os.listdir(path_complete):
        text = open(path_complete+file, 'r')
        line = text.readline()
        while(line):
            ls = line.split()
            word = ls[0]
            if len(word) > 1 or word == "a" or word == "i":
                if word != "nan":
                    if "~" not in word:
                        count = ls[1]
                        if results.get(word) == None:
                            results[word] = int(count)
                        else:
                            results[word] = int(results[word]) + int(count)
            line = text.readline()
    return results
                    

def merge_results_no_filters(path, journal, timelapse, grams):
    results = {}
    path_complete = path+journal+"/"+timelapse+"/"+grams+"/"
    for file in os.listdir(path_complete):
        text = open(path_complete+file, 'r')
        line = text.readline()
        while(line):
            ls = line.split()
            word = ls[0]
            count = ls[1]
            if results.get(word) == None:
                results[word] = int(count)
            else:
                results[word] = int(results[word]) + int(count)
            line = text.readline()
    return results


def getTop(top, results):
    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_x[:top]


def writeTopFile(outpath, results):
    file = open(outpath, 'w')
    for r in results:
        file.write(str(r[0])+'\t'+str(r[1])+'\n')
    file.close()

    
def getImportantWords(nTop, results, infile):
    notImportant = open(infile, 'r').read().splitlines()
    notImportant = set(notImportant)

    N = 100
    english = open("english_words/1grams_2008_english_1000_words", "r").read().splitlines()
    for i in range(N):
        notImportant.add(english[i].split("\t")[0])

    common = getCommonWordsInAllJournals("1grams")
    notImportant.update(common)
#    f2 = [x for x in notImportant if x.isalpha()]
#    notImportant = f2
    
    r = []
    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
    for word in sorted_x:
        flag = True
        s = word[0].split("--")
        for w in s:
            if len(w) == 1:
                flag = False
        if flag:
            s = map(lambda x:x.lower(),s)
            tmp = []
            for x in s:
                try:
                    if x.encode('ascii').isalpha():
                        tmp.append(x)
                    else:
                        flag = False
                except:
                    flag = False
#            tmp = [x for x in s if x.isalpha()]
            if flag:
                items = set(tmp)
                if items.isdisjoint(notImportant):
                    r.append(word)
    return r[:nTop]


def getReservadas(ngram, results, reservadas):
    n = int(ngram.replace("grams", ""))
    r = []
    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
    for word in sorted_x:
        s = word[0].split("--")
        if len(s) == n:
            for element in s:
                if element in reservadas:
                    r.append(word)
                    break
    return r
        

def palabrasReservadas(infile):
    reservadas = open(infile, 'r').read().splitlines()
    path = "results/"
    outpath = "palabras_reservadas/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for journal in os.listdir(path):
        print(journal)
        if journal != "perArticle":
            timelapse = "12months"
#            for timelapse in os.listdir(path+"/"+journal):
            for ngrams in os.listdir(path+"/"+journal+"/"+timelapse):
#            ngrams = "1grams"
                filename = outpath+"reservadas_"+journal+"_"+ngrams+".txt"
#                    filename = outpath+"reservadas_"+journal+"_"+timelapse+"_"+ngrams+".txt"
                if not os.path.isfile(filename):
                    results = merge_results_no_filters(path, journal, timelapse, ngrams)
                    top = getReservadas(ngrams, results, reservadas)
                    writeTopFile(filename, top)
                    print("Filename " + filename)

def palabrasPorRevista():
    nTop = 1000

    path = "/storage/gershenson_g/puig/aps/results/"
    outpath = "/storage/gershenson_g/puig/aps/wordsPerJournal/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for journal in os.listdir(path):
        print(journal)
        if journal != "perArticle":
            timelapse = "12months"
#                for timelapse in os.listdir(path+"/"+journal):                                                                                                                     \
                                                                                                                                                                                     
            for ngrams in os.listdir(path+"/"+journal+"/"+timelapse):
                filename = outpath+"words_"+journal+"_"+ngrams+".txt"

                if not os.path.isfile(filename):
                    results = merge_results(path, journal, timelapse, ngrams)
                    res = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
                    res = res[:nTop]
                    writeTopFile(filename, res)
                    print("Filename " + filename)

                                                                                                                                                                
def palabrasImportantesPorRevista(infile):
#    nTop = 1000
    nTop = 3000000
#    reservadas = open(infile, 'r').read().splitlines()
    path = "results/"
    outpath = "importantWordsPerJournal/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for journal in os.listdir(path):
        print(journal)
        if journal != "perArticle":
            timelapse = "12months"
#            for timelapse in os.listdir(path+"/"+journal):
            for ngrams in os.listdir(path+"/"+journal+"/"+timelapse):
                filename = outpath+"importantes_"+journal+"_"+ngrams+".txt"
#                        filename = outpath+"reservadas_"+journal+"_"+timelapse+"_"+ngrams+".txt"
                if not os.path.isfile(filename):
                    results = merge_results(path, journal, timelapse, ngrams)
                    top = getImportantWords(nTop, results, infile)
                    writeTopFile(filename, top)
                    print("Filename " + filename)


def palabrasImportantes(infile):
    nTop = 3000000
    path = "results/"
    outpath = "importantWords/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for journal in os.listdir(path):
        print(journal)
        if journal != "perArticle":
            for timelapse in os.listdir(path+"/"+journal):
                for ngrams in os.listdir(path+"/"+journal+"/"+timelapse):
                    filename = outpath+"importantes_"+journal+"_"+timelapse+"_"+ngrams+".txt"
                    if not os.path.isfile(filename):
                        results = merge_results(path, journal, timelapse, ngrams)
                        top = getImportantWords(nTop, results, infile)
                        writeTopFile(filename, top)
                        print("Filename " + filename)
                                                                                                                                                                

def getUncommonWordsInJournals(ngram):
    Ntop = 100
    path = "importantWords/"
#    path = "importantWordsPerJournal/"
    outpath = "uncommonWordsPerJournal/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    # get Ntop words in all journals
    for arch in glob.glob(path+'*'+ngram+'*.txt'):
        print(arch)
        print("----------")
        all_words = []
        common = []
        uncommon = []
        top = []
        top = open(arch, 'r').read().splitlines()

        for rest in glob.glob(path+'*'+ngram+'*.txt'):
            print(rest)
            if arch != rest:
                words = []
                words = open(rest, 'r').read().splitlines()
                words = words[:Ntop]
                all_words.append(words)
                
        for t in top:
            flag = True
            f1 = t.split("\t")[0]
            for l in all_words:
                f2 = [w.split('\t', 1)[0] for w in l]
                if f1 in f2:
                    flag = False
            if flag:
                uncommon.append(t)
            else:
                common.append(t)
#        print(common)
#        print("**************")
#        print(uncommon)
                                                            

def getUncommonWordsInAllJournals(ngram):
#    Ntop = 100
    path = "importantWordsPerJournal/"
    outpath = "uncommonWords/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for arch in glob.glob(path+'*'+ngram+'*.txt'):
        name = arch.split("_")
#        print(arch)
#        print("----------")
        all_words = []
        common = []
        uncommon = []
        top = []
        top = open(arch, 'r').read().splitlines()
        
        for rest in glob.glob(path+'*'+ngram+'*.txt'):
#            print(rest)
            if arch != rest:
                words = []
                words = open(rest, 'r').read().splitlines()
#                words = words[:Ntop]
                all_words.append(words)
                
        for t in top:
            flag = True
            f1 = t.split("\t")[0]
            for l in all_words:
                f2 = [w.split('\t', 1)[0] for w in l]
                if f1 in f2:
                    flag = False
            if flag:
                uncommon.append(t)
            else:
                common.append(t)
#        print(uncommon)
        filename = outpath+"uncommon_"+name[1]+"_"+name[2]
        file = open(filename, 'w')
        for r in uncommon:
            file.write(r+'\n')
        file.close()
        print(filename)


        
def getCommonWordsInAllJournals(ngram):
#    nTop = 100
    path = "importantWordsPerJournal1/"
#    outpath = "commonWordsPerJournal/"
#    if not os.path.exists(outpath):
#        os.makedirs(outpath)
        
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

                                                                                                                                                        
def main():
    nTop = 1000
    path = "results/"
    outpath = "top_results/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for journal in os.listdir(path):
        print(journal)
        if journal != "perArticle":
            for timelapse in os.listdir(path+"/"+journal):
                for ngrams in os.listdir(path+"/"+journal+"/"+timelapse):
                    filename = outpath+"top_"+journal+"_"+timelapse+"_"+ngrams+".txt"
                    if not os.path.isfile(filename):
                        results = merge_results(path, journal, timelapse, ngrams)
                        top = getTop(nTop, results)
                        print("Filename " + filename)
                        writeTopFile(filename, top)

if __name__ == "__main__":
#main()
#palabrasReservadas("palabrasReservadas.txt")
    palabrasImportantesPorRevista("functional_words")
    palabrasPorRevista()
#palabrasImportantes("functional_words")

#print(getCommonWordsInAllJournals("1grams"))

#    grams = ["1grams", "2grams", "3grams", "4grams", "5grams", "6grams"]
#    for i in grams:
#        getUncommonWordsInAllJournals(i)
