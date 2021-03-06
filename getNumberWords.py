import os

journals_servers = {"pra": {"name":"Physical_Review_A", "subdir":"/pra/pdf/", "abbr":"Phys. Rev. A."},
                    "prb": {"name":"Physical_Review_B", "subdir":"/prb/pdf/", "abbr":"Phys. Rev. B."},
                    "prc": {"name":"Physical_Review_C", "subdir":"/prc/pdf/", "abbr":"Phys. Rev. C."},
                    "prd": {"name":"Physical_Review_D", "subdir":"/prd/pdf/", "abbr":"Phys. Rev. D."},
                    "pre": {"name":"Physical_Review_E", "subdir":"/pre/pdf/", "abbr":"Phys. Rev. E."},
                    "prl": {"name":"Physical_Review_Letters", "subdir":"/prl/pdf/", "abbr":"Phys. Rev. Lett."},
                    "rmp": {"name":"Review_Modern_Physics", "subdir":"/rmp/pdf/", "abbr":"Rev. Mod. Phys."},
                    "prx": {"name":"Physical_Review_X", "subdir":"/prx/pdf/", "abbr":"Phys. Rev. X."}}

outfile = "totalWords_important.txt"
path = "../../results_important/"
#outfile = "totalWords.txt"
#path = "results/"
month = "12months/"
gram = "1grams/"

words = set()
total = {}

for journal in os.listdir(path):
    if journal != "perArticle":
        files = path + journal + "/" + month + gram
        print(files)
        for f in os.listdir(files):
            f = open(files+f, 'r')
            line = f.readline()
            while line:
                l = line.split("\t")[0]
#                if l.isalpha() and len(l)>1 :
                words.add(l)
                line = f.readline()
            f.close()                                        
        total[journal] = len(words)


f = open(outfile,'w')
keylist = total.keys()
keylist = sorted(keylist)
for key in keylist:
    f.write(key + "\t" + str(total[key]) + "\n")
f.close()


