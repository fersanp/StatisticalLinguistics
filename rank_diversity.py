import pandas as pd
import numpy as np
import os

lapse = ["1months", "3months", "6months", "9months", "12months", "36months", "60months"]
aps = ["pra", "prb", "prc", "prd", "pre", "prl", "rmp", "prx"]
number_of_ngrams=1000
path = "results_important/"
#path = "results/"
outpath = "results_rank_important/"
#outpath = "results_rank/"

for i in range(1,7):
    for journal in aps:
        for timelapse in lapse:
            cpath = path+journal+'/'+timelapse+"/"+str(i)+"grams/"
            ngrams_at_rank=[[0] for i in range(number_of_ngrams)]
            number_of_days=0

            for day in os.listdir(cpath):
                print(cpath)
                print(cpath+day)
    
                df=pd.read_csv(cpath+day,sep='\t',names=['ngram','frequency'])
        
        # from the data frame create a list of words
                ngrams=df['ngram'].tolist()
                if len(ngrams)>number_of_ngrams:
                    for r in range(number_of_ngrams):
                        ngram=ngrams[r]
                        ngrams_at_rank[r].append(ngram)
                    number_of_days=number_of_days+1

            if number_of_days != 0:
                rank_diversity=[len(set(ngrams))/number_of_days for ngrams in ngrams_at_rank]
            
            p = outpath+journal+"/"+timelapse+"/"
            if not os.path.exists(p):
                os.makedirs(p)
            output_file = open(p+str(i)+'grams_RD.txt', 'w')
            for n in range(number_of_ngrams):
                str1=str(n+1)+'\t'+str(rank_diversity[n])
                n=n+1
                str1=str1+'\n' 
                output_file.write(str1)
            output_file.close()

            
