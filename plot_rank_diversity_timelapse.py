import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from RankDiversity import plot_rankDiv
import re

name = {"pra":"A", "prb":"B", "prc":"C", "prd":"D", "pre":"E", "prl":"Lett.", "rmp":"M", "prx":"X"}
lapse = ["1months", "3months", "6months", "9months", "12months", "36months", "60months"]


for n in range(1,7):
    for timelapse in lapse:
        fig = plt.figure()
        colors = itertools.cycle(["r", "b", "g", "c", "m", "#8c564b", "k", "#ff7f0e"])
        for journal in os.listdir('results_rank/'):
            if journal == ".DS_Store":
                continue

#            df=pd.read_csv('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt',sep='\t',names=['rank','diversity'])
            diversity = np.loadtxt('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt')
#            diversity_mean = RD.GetMeanData(diversity)
#            print('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt')
#            print(df.head())
    
#            rank=df['rank'].tolist()
#            diversity=df['diversity'].tolist()
            ax = fig.add_subplot(111)
            if journal == "rmp":
                plot_rankDiv(ax,diversity,colour=next(colors),labels="Rev. Mod. Phys.")
            else:
                plot_rankDiv(ax,diversity,colour=next(colors),labels="Phys. Rev. "+name[journal])
            

#            fig.savefig("~/foo.png")
#            plt.scatter(rank,diversity,s=2,label="PhysRev "+name[journal])
#            p30 = np.poly1d(np.polyfit(rank, diversity, 10))
#            plt.plot(rank, p30(rank), label="PhysRev "+name[journal])

#        plt.xlim([1,1000])
#        plt.xscale('log')
        plt.xlabel('$k$',size=16)
        plt.ylabel('$d(k)$',rotation='horizontal',labelpad=25,size=16)
        parse = re.split('(\d+)',timelapse)
        if timelapse == "1months":
            plt.title("Rank diversity: "+str(n)+'grams'+'\n'+"Time interval: "+parse[1]+" "+parse[2][:-1],size=16)
        else:
            plt.title("Rank diversity: "+str(n)+'grams'+'\n'+"Time interval: "+parse[1]+" "+parse[2],size=16)
#        plt.xticks(size=15)
#        plt.yticks(size=15)
#        plt.tight_layout()
        plt.legend(loc=4,prop={'size':9})
#        plt.savefig('graphs_important/physical_rank_diversities_'+timelapse+"_fit_witherrors"+str(n)+'grams'+'.png')#,dpi=256)
        outpath = "/home/fer/aps/graphs_important/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(outpath+'/physical_rank_diversities_'+timelapse+"_"+str(n)+'grams'+'.png')#,dpi=256)
