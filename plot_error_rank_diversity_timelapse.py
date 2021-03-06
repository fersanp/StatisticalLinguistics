import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import RankDiversity
from RankDiversity import plot_rankDiv
import re
from scipy import stats


name = {"pra":"A", "prb":"B", "prc":"C", "prd":"D", "pre":"E", "prl":"Lett.", "rmp":"M", "prx":"X"}
lapse = ["1months", "3months", "6months", "9months", "12months", "36months", "60months"]


for n in range(1,7):
    for timelapse in lapse:
        fig = plt.figure()
        colors = itertools.cycle(["r", "b", "g", "c", "m", "#8c564b", "k", "#ff7f0e"])
        for journal in os.listdir('results_rank/'):
            journal = "pra"
            if journal == ".DS_Store":
                continue

            df=pd.read_csv('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt',sep='\t',names=['rank','diversity'])
#            diversity = df["diversity"]
            diversity = np.loadtxt('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt')
            diversity_mean = RankDiversity.GetMeanData(diversity)
#            print('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt')
#            print(df.head())
#            print(journal)
#            print(diversity)
#            print(journal)
#            print(diversity_mean)
#            print(diversity_mean[:,0])
#            print(diversity_mean[:,1])
#            print("--------------------")
#            rank=df['rank'].tolist()
#            diversity=df['diversity'].tolist()
            ax = fig.add_subplot(111)
            if journal == "rmp":
#                plt.errorbar(diversity_mean[:,0],diversity_mean[:,1],color=next(colors),label="Rev. Mod. Phys.",
#                             yerr=stats.sem(diversity[:,1]), ecolor='r')
                plot_rankDiv(ax,diversity,showFit=True,colour=next(colors),labels="Rev. Mod. Phys.")
                plt.scatter(df["rank"].tolist(), df["diversity"].tolist())
            else:
#                plt.errorbar(diversity_mean[:,0],diversity_mean[:,1],color=next(colors),label="Phys. Rev. "+name[journal],
#                             yerr=stats.sem(diversity[:,1]), ecolor='r')
                plot_rankDiv(ax,diversity,showFit=True,colour=next(colors),labels="Phys. Rev. "+name[journal])
                plt.scatter(df["rank"].tolist(), df["diversity"].tolist())
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
        outpath = "/home/fer/aps/graphs_error/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(outpath+'physical_rank_diversities_'+timelapse+"_fit_witherrors"+str(n)+'grams'+'.png')#,dpi=256)

