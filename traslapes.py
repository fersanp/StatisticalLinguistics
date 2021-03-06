import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas
import sys

#grams = [1,2,3,4,5,6]
grams = [1]
journals = ['pra', 'prb', 'prc', 'prd', 'pre', 'prl', 'prx', 'rmp']
ticks = ["Phys. Rev. A", "Phys. Rev. B", "Phys. Rev. C", "Phys. Rev. D", "Phys. Rev. E", "Phys. Rev. L", "Phys. Rev. X", "Rev. Mod. Phys."]
journals_anios = {'pra':{'start_year':1970, 'start_volume':1, 'end_year':2017, 'end_volume':96, 'timelapse':6, "abbr":"Phys. Rev. A"},
                  'prb':{'start_year':1970, 'start_volume':1, 'end_year':2017, 'end_volume':96, 'timelapse':6, "abbr":"Phys. Rev. B"},
                  'prc':{'start_year':1970, 'start_volume':1, 'end_year':2017, 'end_volume':96, 'timelapse':6, "abbr":"Phys. Rev. C"},
                  'prd':{'start_year':1970, 'start_volume':1, 'end_year':2017, 'end_volume':96, 'timelapse':6, "abbr":"Phys. Rev. D"},
                  'pre':{'start_year':1993, 'start_volume':47, 'end_year':2017, 'end_volume':96, 'timelapse':6, "abbr":"Phys. Rev. E"},
                  'prl':{'start_year':1959, 'start_volume':2, 'end_year':2017, 'end_volume':119, 'timelapse':6, "abbr":"Phys. Rev. L"},
                  'prx':{'start_year':2012, 'start_volume':2, 'end_year':2017, 'end_volume':7, 'timelapse':12, "abbr":"Phys. Rev. X"},
                  'rmp':{'start_year':1930, 'start_volume':2, 'end_year':2017, 'end_volume':89, 'timelapse':12, "abbr":"Rev. Mod. Phys."}}


def getVolumeNumber(journal, year):
    start = int(journals_anios[journal]['start_year'])
    start_volume = int(journals_anios[journal]['start_volume'])
    timelapse = int(journals_anios[journal]['timelapse'])
    n = (year - start) * (12//timelapse) + start_volume
    return n

    
def overlapsPerYear(top):
    percentage_functional = {}
    percentage_content = {}
    arch_functional_words = open("functional_words").read().splitlines() 
    for i in range(len(journals)):
        journal = journals[i]
        percentage_functional[journal] = {}
        percentage_content[journal] = {}
        start_year = journals_anios[journal]['start_year']
        years = range(int(start_year), 2017)
        path = 'results_important/'
        for year in years:
            percentage_functional[journal][year] = {}
            percentage_content[journal][year] = {}
            n = getVolumeNumber(journal, year)
            arch = path+journal+'/12months/'+'1grams/' + str(n) + "_January"
            exists = os.path.isfile(arch)
            if (exists):
                text = [x.split('\t')[0] for x in open(arch).readlines()][:top]
                print(arch)
                print(len(text))

                for j in range(len(journals)):
                    other_journal = journals[j]
                    if journal != other_journal:
                        n = getVolumeNumber(other_journal, year)
                        arch = path+ other_journal+'/12months/'+'1grams/' + str(n) + "_January"
                        exists = os.path.isfile(arch)
                        if (exists):
                            other_text = [x.split('\t')[0] for x in open(arch).readlines()][:top]
                            
                            total_count = len(text)
                            functional_words_count = 0
                            content_words_count = 0
                            for word in other_text:
                                if len(word) > 1 or word == "a" or word == "i":
                                    if word != "nan":
                                        if "~" not in word:
                                            if word in arch_functional_words: 
                                                functional_words_count += 1
#                                else:
                                            if word in text:
                                                content_words_count += 1

                            percentage_funct_words = functional_words_count/total_count
                            percentage_content_words = content_words_count/total_count
                            percentage_functional[journal][year].update({other_journal:percentage_funct_words})
                            percentage_content[journal][year].update({other_journal:percentage_content_words})

    return (percentage_functional, percentage_content)

                                                                                                                                            
def plotOverlapsPerYear(percentage_functional, percentage_content, journal_reference):
    plt.figure()
    data_func = percentage_functional[journal_reference]
    df_func = pandas.DataFrame(data_func)
    data_cont = percentage_content[journal_reference]
    df_cont = pandas.DataFrame(data_cont)
#    print(df_cont)
    x_func = list(df_func.columns.values)
    x_cont = list(df_cont.columns.values)
#    for index, row in df_func.iterrows():
#        plt.plot(x_func,row, label=index)

    for index, row in df_cont.iterrows():
        plt.plot(x_cont, row, label=journals_anios[index]["abbr"])
#        plt.plot(x_cont,row, '--')

#    df_mean = df_cont.mean(axis=0, skipna = True)
#    print(df_mean)
#    plt.plot(df_mean.index, df_mean.values)

    plt.title("Overlaps per year with respect to "+journal_reference)
    plt.xlabel("year")
    plt.ylabel("Overlap")
    plt.legend(prop={'size':8})
    plt.savefig("overlaps/overlaps_important.png")
    
                                                                                            
# Porcentaje de palabras en las otras revistas para todos los grams (matrix plot)
def porcentage_overlaps(top):
    percentage_funct = {}
    for ngram in grams:
        percentage_funct[ngram] = {}
        for i in range(len(journals)):
            journal = journals[i]
            percentage_funct[ngram][journal] = {}
            if only_content_words:
                path = 'importantWordsPerJournal/'
                arch = path+'importantes_'+journal+"_"+str(ngram)+'grams.txt'
            else:
                path = 'wordsPerJournal/'
                arch = path+'words_'+journal+"_"+str(ngram)+'grams.txt'
            text = [x.split('\t')[0] for x in open(arch).readlines()][:top]

            for j in range(len(journals)):
                other_journal = journals[j]
                if journal == other_journal:
                    percentage_funct[ngram][journal].update({other_journal:1})
                else:
                    if only_content_words:
                        arch = path+'importantes_'+other_journal+"_"+str(ngram)+'grams.txt'
                    else:
                        arch = path+'words_'+other_journal+"_"+str(ngram)+'grams.txt'
                    other_text = [x.split('\t')[0] for x in open(arch).readlines()][:top]
                
                    total_count = len(text)
                    functional_words_count = 0
                    for word in other_text:
                        if word in text:
                            functional_words_count += 1
                    percentage = functional_words_count/total_count
                    percentage_funct[ngram][journal].update({other_journal:percentage})
    return percentage_funct


def matrixplot_overlaps(overlaps,top):
    for gram in overlaps.keys():
        df = pandas.DataFrame(overlaps[gram])
        df = df.values
        fig, ax = plt.subplots()
#        im = ax.matshow(df, cmap=plt.cm.hot, vmin=0, vmax=1)
#        ax.figure.colorbar(im)
#        ax.set_xticks(np.arange(len(journals)))
#        ax.set_yticks(np.arange(len(journals)))
#        ax.set_xticklabels(ticks, fontsize=8, rotation=45)
#        ax.set_yticklabels(ticks, fontsize=8)

    # Plot the heatmap
        im = ax.matshow(df, cmap="YlGn", vmin=0, vmax=1)
        for (i, j), z in np.ndenumerate(df):
            if i == j:
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color="w")
            else:
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")

    # We want to show all ticks...
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_yticks(np.arange(df.shape[0]))
    # ... and label them with the respective list entries.
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)

    # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

    # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(df.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(df.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        fig.tight_layout()

        outpath = "overlaps/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        if only_content_words:
            fig.suptitle("Overlaps Content Words: "+str(gram)+"grams. "+"Top "+str(top)+" words", fontsize=9)
#            ax.set_title("Overlaps Content Words: "+str(gram)+"grams\n"+"Top "+str(top)+" words", fontsize=9)
            plt.savefig(outpath+"overlaps_content_matrix_plot_"+str(gram))
        else:
            fig.suptitle("Overlaps Functional Words: "+str(gram)+"grams. Top "+str(top)+" words", fontsize=9)
#            ax.set_title("Overlaps Functional Words: "+str(gram)+"grams\nTop "+str(top)+" words", fontsize=9)
            plt.savefig(outpath+"overlaps_functional_matrix_plot_"+str(gram))



def overlaps_mean(overlaps):
    mean = []
    for gram in overlaps.keys():
        df = pandas.DataFrame(overlaps[gram])
        df = df.values
        m = df.shape[0]
        r,c = np.triu_indices(m,1)
        mean.insert(gram, np.mean(df[r,c]))
    return mean



def plot_overlaps_mean(overlaps_mean, n):
    plt.figure()
    if only_content_words:
        plt.title("Mean of Overlaps: Content Words ")
    else:
        plt.title("Mean of Overlaps: Functional Words ")
    plt.xlabel("n-Grams")
    plt.ylabel("Mean")

    lists = sorted(overlaps_mean.items()) # sorted by key, return a list of tuples
    for i in lists:
        plt.plot(range(1,len(i[1])+1), i[1], label=journal_anios[i[0]]["abbr"], marker="o")

    plt.legend(prop={'size':8})
    plt.savefig("overlaps/overlaps_mean.png")
    


def plot_overlaps_mean(overlaps_mean_content, overlaps_mean_functional, n):
    print(overlaps_mean_content)
#    plt.figure()
    fig, ax = plt.subplots()
    plt.title("Mean of Overlaps")
    plt.xlabel("n-Grams")
    plt.ylabel("Mean")


    cont = []
    func = []

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    listsC = sorted(overlaps_mean_content.items()) 
    listsF = sorted(overlaps_mean_functional.items()) 
    for i in range(len(listsC)):
        data = listsC[i][1]
        print(data)
        cont  += ax.plot(range(1,len(data)+1), data, linestyle='-', color=colors[i], marker="o", markersize=6, label=listsC[i][0])
        print()
        data = listsF[i][1]
        print(data)
        func += ax.plot(range(1,len(data)+1), data, linestyle='--', color=colors[i], marker="o", markersize=6)

    category1 = ['Content words', 'Functional words']
#    plt.legend([p1, p2], category1)
#    ax.add_artist(legend1)

    ax.legend(prop={'size':8}, loc='upper right', title="Top words")

    from matplotlib.legend import Legend
    leg = Legend(ax, [cont[0], func[0]], category1, loc='lower left', handlelength=4, prop={'size':8})
    ax.add_artist(leg)

    plt.savefig("overlaps/overlaps_mean.png")

        
top = 100
maxTop = 1100
#######  Create heat maps #######

only_content_words = False
r = porcentage_overlaps(top)
matrixplot_overlaps(r, top)

#only_content_words = True
#r = porcentage_overlaps(top)
#matrixplot_overlaps(r, top)

sys.exit(1)
#######  Mean plots  ########
mC = {}
n = range(top, maxTop, 200)
only_content_words = True
for i in n:
    r = porcentage_overlaps(i)
    print("mean content"+str(i))
    mC[i] = overlaps_mean(r)
    print("aqui "+str(i))
#plot_overlaps_mean(mC, n)


mF = {}
only_content_words = False
for i in n:
    r = porcentage_overlaps(i)
    mF[i] = overlaps_mean(r)
    print("aca "+str(i))

plot_overlaps_mean(mC, mF, n)




#o = overlapsPerYear(top)
#plotOverlapsPerYear(o[0], o[1], "rmp")

