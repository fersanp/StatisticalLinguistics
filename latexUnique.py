import os
import glob
import subprocess

def createTable(title, ntop, files):
    keylist = files.keys()
    keylist = sorted(keylist)
    ncolumns = len(files)
    colpos = ""
    for i in range(ncolumns):
        colpos += "|p{2.2cm} |p{1cm} "
    colpos += "|"

    headers = "\\centering \n" \
              "\\footnotesize \n" \
              "\\setlength\LTleft{-1.7in} \n" \
              "\\setlength\LTright{-1.7in} \n" \
              "\\begin{longtable}{"+colpos+"} \n" \
              "\\caption{"+title+"}  \n" \
              "\\\\ \\hline \n"

    h = ""
    for hi in keylist:
        h += "\multicolumn{2}{|c|}{"+hi+"}" + "&"
    h = h[:-1]
    h += " \\\\ \n \\hline \n"

    
    lines = []
    for f in keylist:
        lines.append(open(files[f],'r').readlines())
        
    j = 0
    rows = ""
    for i in range(ntop):
        for j in range(len(lines)):
            print(j)
            word = lines[j][i].split()
            print(word)
            rows += word[0] + "&" + word[1][:-1] + "&"
        rows = rows[:-1]
        rows += "\\\\ \n  \\hline \n"
        
    closure = "\\end{longtable} \n" 

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

ntop = 10
#grams = ["1","2","3","4","5","6"]
grams = ["1"]
##months = ["1","3","6","9","12","24","36","48","60"]
months = ["1","3","6","9","12","36","60"]
journals = ["pra", "prb", "prc", "prd", "pre", "prx", "rmp", "prl", "prx", "rmp"]
path = "uniqueWords/"
#path = "importantWordsPerJournal/"
outpath = "sorted_unique_tables/"
for gram in grams:
#    english = glob.glob("/home/fer/aps/english_words/"+gram+"grams*")
#    for month in months:
    files = {}
#        files["English"] = english[0]
    for archive in glob.glob(path+'*'+gram+'grams.txt'):
        print(archive)
        head = archive.split("_")
        if "pra" in archive or "prb" in archive or "prc" in archive or "prd" in archive or "pre" in archive:
            files[journals_servers[head[1]]["abbr"]] = archive
            print(files)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outputfile = outpath + "table_sorted_"+str(ntop)+"_"+gram+"grams.tex"
#    outputfile = outpath + "table_sorted_"+month+"months_"+gram+"grams.tex"
#        title = "Top "+str(ntop)+"\\\\" + "Rank Diversity: "+gram+"grams \\\\" + "Time Interval: "+month+"months"
    title = "Top "+str(ntop)+" Unique Content Words\\\\" + "Rank Diversity: "+gram+"grams"

    st = "\\documentclass{article} \n" \
         "\\usepackage{caption}" \
         "\\usepackage[english]{babel} \n" \
         "\\usepackage{longtable} \n" \
         "\\usepackage{pdflscape} \n" \
         "\\usepackage{tabularx} \n" \
         "\\begin{document} \n"

    st += createTable(title, ntop, files)
    st += "\\end{document}"
    print(st)
    out = open(outputfile, 'w')
    out.write(st)
    out.close()

    convert2PDF(outpath, outputfile)
    print("*******************")


######################
#ntop = 10
for gram in grams:
    files = {}
    for archive in glob.glob(path+'*'+gram+'grams.txt'):
        print(archive)
        head = archive.split("_")
        if  "prl" in archive or "prx" in archive or "rmp" in archive:
            files[journals_servers[head[1]]["abbr"]] = archive
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outputfile = outpath + "table2_sorted_"+str(ntop)+"_"+gram+"grams.tex"
    title = "Top "+str(ntop)+" Unique Content Words\\\\" + "Rank Diversity: "+gram+"grams"
            
    st = "\\documentclass{article} \n" \
         "\\usepackage{caption} \n" \
         "\\usepackage[english]{babel} \n" \
         "\\usepackage{longtable} \n" \
         "\\usepackage{pdflscape} \n" \
         "\\usepackage{tabularx} \n" \
         "\\begin{document} \n"
    
    st += createTable(title, ntop, files)
    st += "\\end{document}"
    
    out = open(outputfile, 'w')
    out.write(st)
    out.close()
    
    convert2PDF(outpath, outputfile)
