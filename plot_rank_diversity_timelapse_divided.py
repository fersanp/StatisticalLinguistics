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
        colors = itertools.cycle(["b", "c", "m", "#8c564b", "#ff7f0e"])
        for journal in os.listdir('results_rank/'):

            if not(journal == ".DS_Store") and not(journal == "prl") and not(journal == "prx") and not(journal == "rmp"):
                diversity = np.loadtxt('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt')

                ax = fig.add_subplot(111)
                if journal == "rmp":
                    plot_rankDiv(ax,diversity,colour=next(colors),labels="Rev. Mod. Phys.")
                else:
                    plot_rankDiv(ax,diversity,colour=next(colors),labels="Phys. Rev. "+name[journal])
            
                plt.xlabel('$k$',size=16)
                plt.ylabel('$d(k)$',rotation='horizontal',labelpad=25,size=16)
                parse = re.split('(\d+)',timelapse)
                if timelapse == "1months":
                    plt.title("Rank diversity: "+str(n)+'grams'+'\n'+"Time interval: "+parse[1]+" "+parse[2][:-1],size=16)
                else:
                    plt.title("Rank diversity: "+str(n)+'grams'+'\n'+"Time interval: "+parse[1]+" "+parse[2],size=16)
        plt.legend(loc=4,prop={'size':9})

        outpath = "/home/fer/aps/graphs_separated/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        plt.savefig(outpath+'/physical_rank_diversities_'+timelapse+"_separated_"+str(n)+'grams'+'.png')

        
#########################
for n in range(1,7):
    for timelapse in lapse:
        fig = plt.figure()
        colors = itertools.cycle(["r", "g", "k"])
        for journal in os.listdir('results_rank/'):

            if not(journal == ".DS_Store") and not(journal == "pra") and not(journal == "prb") and not(journal == "prc") and not(journal == "prd") and not(journal == "pre") :
                diversity = np.loadtxt('results_rank/'+journal+"/"+timelapse+"/"+str(n)+'grams_RD.txt')
                
                ax = fig.add_subplot(111)
                if journal == "rmp":
                    plot_rankDiv(ax,diversity,colour=next(colors),labels="Rev. Mod. Phys.")
                else:
                    plot_rankDiv(ax,diversity,colour=next(colors),labels="Phys. Rev. "+name[journal])

                plt.xlabel('$k$',size=16)
                plt.ylabel('$d(k)$',rotation='horizontal',labelpad=25,size=16)
                parse = re.split('(\d+)',timelapse)
                if timelapse == "1months":
                    plt.title("Rank diversity: "+str(n)+'grams'+'\n'+"Time interval: "+parse[1]+" "+parse[2][:-1],size=16)
                else:
                    plt.title("Rank diversity: "+str(n)+'grams'+'\n'+"Time interval: "+parse[1]+" "+parse[2],size=16)
        plt.legend(loc=4,prop={'size':9})
        
        outpath = "/home/fer/aps/graphs_separated/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        plt.savefig(outpath+'/physical_rank_diversities_'+timelapse+"_separated_LXR_"+str(n)+'grams'+'.png')#,dpi=256)

                                                                                                                                                                
