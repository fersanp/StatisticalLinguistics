import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import itertools

path = "importantWordsPerJournal/"
outpath = "wordsProbability/"
journals = ["pra","prb","prc","prd","pre","prx","rmp","prl"]
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown'])

totalWords = {}
 # get Number words in all journals
importantes = open("totalWords.txt", 'r').read().splitlines()  
for line in importantes: 
    journal = line.split("\t")[0]
    total = line.split("\t")[1]
    totalWords[journal] = int(total)

print(totalWords)

                           
for ngram in range(1,7):
    plt.figure()
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown'])
    for arch in glob.glob(path+'*'+str(ngram)+'grams*.txt'):
        prob = []
        print(arch)
        journal = arch.split("_")[1]
        print(journal)
    
        outfilename = outpath+"probability_"+str(ngram)+"grams.png"
        if not os.path.isfile(outfilename):
            importantes = open(arch, 'r').read().splitlines()
            for w in importantes:
                frec = float(w.split("\t")[1])
                prob.append(frec/totalWords[journal])

        plt.loglog(prob, label=journal, color=next(colors))
    plt.xlabel('Rank (log)')
    plt.ylabel('Probability (log)')
    plt.legend(fontsize='small')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    plt.savefig(outfilename)
