import os
import glob
import subprocess
from collections import Counter
import random
import numpy as np


def colors(n): 
    ret = [] 
    r = int(random.random() * 256) 
    g = int(random.random() * 256) 
    b = int(random.random() * 256) 
    step = 256 / n 
    for i in range(n): 
        r += step 
        g += step 
        b += step 
        r = int(r) % 256 
        g = int(g) % 256 
        b = int(b) % 256 
        ret.append((str(r),str(g),str(b)))  
    return ret 


def commonWords(start, ntop, files):
    keylist = files.keys()
    keylist = sorted(keylist)

    lines = []
    for f in keylist:
        lines.append(open(files[f],'r').readlines())

    words = []
    for i in range(start, ntop):
        for j in range(len(files)):
            l = lines[j][i].split()
            words.append(l[0])

    s = Counter(words)
    commons = [k for k,v in s.items() if v == len(journals)]
    return commons


def createTable(title, start, ntop, files):
    keylist = files.keys()
    keylist = sorted(keylist)
    ncolumns = len(files)
    colpos = "|c "
    for i in range(ncolumns):
        colpos += "|p{2cm} "
    colpos += "| "

    headers = "\\centering \n" \
              "\\footnotesize \n" \
              "\\setlength\LTleft{-1.7in} \n" \
              "\\setlength\LTright{-1.7in} \n" \
              "\\begin{longtable}{"+colpos+"} \n" \
              "\\caption*{"+title+"} \\\\ \n" \
              "\\hline\\hline \n"

    h = "Rank & \n"
    for hi in range(len(keylist)-1):
        h += str(keylist[hi]) + " & "
    h += keylist[-1]
    h += "\\\\ \n \\hline \\hline \n"


    commons = commonWords(start, ntop, files)
    if commons:
        whites = np.linspace(20, 90, len(commons))

    lines = []
    for f in keylist:
        lines.append(open(files[f],'r').readlines())

    j = 0
    rows = ""
    for i in range(start, ntop):
        rows += str(i+1) + " & " 
        for j in range(len(files)-1):
            word = lines[j][i].split()
            if word[0] in commons:
                rows += "\cellcolor{gray!"+str(int(whites[commons.index(word[0])]))+"}" + word[0] + " & "
#                rows += "\cellcolor[rgb]{"+",".join(color[commons.index(word[0])])+"}" + word[0] + "&"
            else:
                rows += word[0] + " & "
        last = lines[j+1][i].split()[0]
        if last in commons:
            rows += "\cellcolor{gray!"+str(int(whites[commons.index(last)]))+"}" + last 
#            rows += "\cellcolor{gray!40}" + last 
        else:
            rows += last
        rows += "\\\\ \n \\hline \n"
        
    closure = "\\hline \n" \
              "\\end{longtable} \n" 

    final = headers + h + rows + closure
    return final


def convert2PDF(outpath, article):
    print("PDF "+article)
    comm = "pdflatex -output-directory " + outpath + " " + article
    res = subprocess.call(['pdflatex', '-output-directory', outpath, article], shell=False)


journals_servers = {"pra": {"name":"Physical_Review_A", "subdir":"/pra/pdf/", "abbr":"Phys. Rev. A"},
                    "prb": {"name":"Physical_Review_B", "subdir":"/prb/pdf/", "abbr":"Phys. Rev. B"},
                    "prc": {"name":"Physical_Review_C", "subdir":"/prc/pdf/", "abbr":"Phys. Rev. C"},
                    "prd": {"name":"Physical_Review_D", "subdir":"/prd/pdf/", "abbr":"Phys. Rev. D"},
                    "pre": {"name":"Physical_Review_E", "subdir":"/pre/pdf/", "abbr":"Phys. Rev. E"},
                    "prl": {"name":"Physical_Review_Letters", "subdir":"/prl/pdf/", "abbr":"Phys. Rev. Lett."},
                    "rmp": {"name":"Review_Modern_Physics", "subdir":"/rmp/pdf/", "abbr":"Rev. Mod. Phys."},
                    "prx": {"name":"Physical_Review_X", "subdir":"/prx/pdf/", "abbr":"Phys. Rev. X"}}


start = 80
ntop = 100
#grams = ["1","2","3","4","5","6"]
grams = ["1"]
#months = ["1","3","6","9","12","24","36","48","60"]
months = ["1","3","6","9","12","36","60"]
#journals = ["pra", "prb", "prc", "prd", "pre", "prx", "rmp", "prl"]
journals = ["pra", "prb", "prc", "prd", "pre"]
path = "importantWordsPerJournal/"
outpath = "sorted_important_tables/"
for gram in grams:
#    english = glob.glob("/home/fer/aps/english_words/"+gram+"grams*")
#    for month in months:
    files = {}
#        files["English"] = english[0]
    for archive in glob.glob(path+'*'+gram+'grams.txt'):
        head = archive.split("_")
        if head[1] in journals:
            print(archive)
            files[journals_servers[head[1]]["abbr"]] = archive
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outputfile = outpath + "table_sorted_"+str(ntop)+"_"+gram+"grams.tex"
#    outputfile = outpath + "table_sorted_"+month+"months_"+gram+"grams.tex"
#        title = "Top "+str(ntop)+"\\\\" + "Rank Diversity: "+gram+"grams \\\\" + "Time Interval: "+month+"months"
    title = "Top "+str(ntop)+" Content Words\\\\" + "Rank Diversity: "+gram+"grams"

    st = "\\documentclass{article} \n" \
         "\\usepackage{rotating} \n" \
         "\\usepackage{caption}" \
         "\\usepackage[english]{babel} \n" \
         "\\usepackage{longtable} \n" \
         "\\usepackage{pdflscape} \n" \
         "\\usepackage{tabularx} \n" \
         "\\usepackage{xcolor,colortbl} \n" \
         "\\begin{document} \n"

    st += createTable(title, start, ntop, files)
    st += "\\end{document}"
    
    out = open(outputfile, 'w')
    out.write(st)
    out.close()

    convert2PDF(outpath, outputfile)
