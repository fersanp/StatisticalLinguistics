import os
import glob
import re
import subprocess


def convert2PDF(outpath, article):
    print("PDF "+article)
    comm = "pdflatex -output-directory " + outpath + " " + article
    res = subprocess.call(['pdflatex', '-output-directory', outpath, article], shell=False)

                
def findFamous():
    path = "importantWordsPerJournal/"
    filename = "famous.txt"
    names = sorted(open(filename,'r').readlines())
    results = {}

    for name in names:
        name = name.rstrip()
        famous = name.split()
        gram = len(famous)
        for archive in glob.glob(path+'*'+str(gram)+'grams.txt'):
            journal = re.search('_(.*)_',archive).group(1)
            found = False
            text = open(archive, 'r')
            line = text.readline()
            j = 1
            while(not found and line):
                ls = line.split()
                word = ls[0]
                if word == '--'.join(famous):
                    found = True
                    if not(name in results):
                        results[name] = {}
                    results[name][journal] = str(j)

                line = text.readline()
                j += 1
    return (results)


def writePDFtable(results):
    outpath = "sorted_famous_tables/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outputfile = outpath + "table_sorted_famous.tex"

    title = "Rank of Famous Physicist"

    st = "\\documentclass{article} \n" \
         "\\usepackage{caption}" \
         "\\usepackage[english]{babel} \n" \
         "\\usepackage{longtable} \n" \
         "\\usepackage{pdflscape} \n" \
         "\\usepackage{tabularx} \n" \
         "\\usepackage{xcolor,colortbl} \n" \
         "\\begin{document} \n"

    keylist = results.keys()
    keylist = sorted(keylist)
    ncolumns = len(journals)
    colpos = "|p{2cm}"
    for i in range(ncolumns):
        colpos += "|p{2cm}"
    colpos += "|"

    headers = "\\centering \n" \
              "\\footnotesize \n" \
              "\\begin{longtable}{"+colpos+"} \n" \
              "\\caption{"+title+"} \\\\ \n" \
              "\\hline \n"


    h = "Physicist&"
    for journal in journals:
        h += journals_servers[journal]["abbr"] + "&"
    h = h[:-1]
    h += "\\\\ \n \\hline \\hline \n"

    rows = ""
    for name in keylist:
        rows += name + " & "
        for journal in journals:
            j = results[name].get(journal)
            if j:
                if int(j) <= 1000:
                    rows += "\cellcolor{gray}" + j + "&"
                else:
                    rows += j + "&"
            else:
                rows += "&"
        rows = rows[:-1]
        rows += "\\\\ \n \\hline \n"

    closure = "\\end{longtable} \n"

    final = headers + h + rows + closure
    st += final + "\\end{document}"

    out = open(outputfile, 'w')
    out.write(st)
    out.close()

    convert2PDF(outpath, outputfile)



journals_servers = {"pra": {"name":"Physical_Review_A", "subdir":"/pra/pdf/", "abbr":"Phys. Rev. A"},
                    "prb": {"name":"Physical_Review_B", "subdir":"/prb/pdf/", "abbr":"Phys. Rev. B"},
                    "prc": {"name":"Physical_Review_C", "subdir":"/prc/pdf/", "abbr":"Phys. Rev. C"},
                    "prd": {"name":"Physical_Review_D", "subdir":"/prd/pdf/", "abbr":"Phys. Rev. D"},
                    "pre": {"name":"Physical_Review_E", "subdir":"/pre/pdf/", "abbr":"Phys. Rev. E"},
                    "prl": {"name":"Physical_Review_Letters", "subdir":"/prl/pdf/", "abbr":"Phys. Rev. Lett."},
                    "rmp": {"name":"Review_Modern_Physics", "subdir":"/rmp/pdf/", "abbr":"Rev. Mod. Phys."},
                    "prx": {"name":"Physical_Review_X", "subdir":"/prx/pdf/", "abbr":"Phys. Rev. X"}}

    
#journals = ["pra", "prb", "prc", "prd", "pre", "prl", "prx", "rmp"]
journals = ["pra", "prb", "prc", "prd", "pre"]
r = findFamous()
print(r)
writePDFtable(r)
