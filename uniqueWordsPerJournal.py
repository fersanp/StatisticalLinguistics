import os
import operator


path = "importantWordsPerJournal/"
journals = ["pra", "prb", "prc", "prd", "pre", "prx", "prl", "rmp"]
#grams = ["1", "2", "3", "4", "5", "6"]
grams = ["1"]


# Palabras que no aparecen en las otras revistas para todos los grams
def no_overlaps(top):
    percentage_funct = {}
    for ngram in grams:
        print(ngram)
        percentage_funct[ngram] = {}
        for i in range(len(journals)):
            journal = journals[i]
            print(journal)
            percentage_funct[ngram][journal] = {}
            arch = path+'importantes_'+journal+"_"+str(ngram)+'grams.txt'
            text = set()
            for x in open(arch).readlines()[:top]: 
                s = x.split("\t")
                t = (s[0], int(s[1][:-1]))
                text.add(t)
            
            for j in range(len(journals)):
                other_journal = journals[j]
                if journal != other_journal:
                    arch = path+'importantes_'+other_journal+"_"+str(ngram)+'grams.txt'
                    other_text = set()
                    for x in open(arch).readlines()[:top]:
                        s = x.split("\t")
                        t = (s[0], int(s[1][:-1]))
                        other_text.add(t)

                    b2 = {n for n,x in other_text}
                    c = {(n,x) for (n,x) in text if n not in b2}
                    text = c

            percentage_funct[ngram][journal] = text
    return percentage_funct

                                                                
#r = no_overlaps(1000000)
r = no_overlaps(1000)
#print(r)
for k, v in r.items():
    for key, value in v.items():
        sorted_dict = sorted(list(value), key=lambda x: x[1], reverse=True)
        with open("uniqueWords/uniqueWords_"+key+"_"+k+"grams.txt", 'w') as f:
            i = 0
            for e in sorted_dict:
                f.write(e[0] + "\t" + str(e[1]) + "\n")

                                
