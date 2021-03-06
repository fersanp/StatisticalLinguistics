import subprocess
import time
import os
import wget
import ast
import PyPDF2
from urllib.request import urlopen
import multiprocessing
import threading

DOI_DIR = "10.1103/"
HOST_PHYSICAL = "https://journals.aps.org"

journals = ["pra", "prb", "prc", "prd", "pre", "prl", "rmp", "prx"]


journals_servers = {"pra": {"name":"Physical_Review_A", "subdir":"/pra/pdf/", "abbr":"PhysRevA."},
                    "prb": {"name":"Physical_Review_B", "subdir":"/prb/pdf/", "abbr":"PhysRevB."},
                    "prc": {"name":"Physical_Review_C", "subdir":"/prc/pdf/", "abbr":"PhysRevC."},
                    "prd": {"name":"Physical_Review_D", "subdir":"/prd/pdf/", "abbr":"PhysRevD."},
                    "pre": {"name":"Physical_Review_E", "subdir":"/pre/pdf/", "abbr":"PhysRevE."},
                    "prl": {"name":"Physical_Review_LETTERS", "subdir":"/prl/pdf/", "abbr":"PhysRevLett."},
                    "rmp": {"name":"Review_Modern_Physics", "subdir":"/rmp/pdf/", "abbr":"RevModPhys."},
                    "prx": {"name":"Physical_Review_X", "subdir":"/prx/pdf/", "abbr":"PhysRevX."}}


#prb.txt
#volume_issues = {'prb': {'91': {'March': [['094101', '099907']],
#                                'January': [['014101', '019903'], ['020101', '024514'], ['035101', '039910'], ['041101', '045442']],
#                                'February': [['054101', '059903'], ['060101', '064513'], ['075101', '079902'], ['081101', '085433']]},
#                         '70': {'September': [['092101', '099904']],
#                                'August': [['052101', '059904'], ['060101', '069903'], ['073101', '079901'], ['081101', '087401']]}}}
volumes_issues = {}

#volumes_years = {'prb': {'91': '2015',
#                        '70': '2004'}}
volumes_years = {}


def dic_volumes(path):
    print("CREATING ISSUES DICTIONARY")
    volume = ""
    for file in os.listdir(path):
        print(file)
        journal = file[:-4]
        text = open(path+file, 'r')
        line = text.readline()
        while(line):
            ls = line.split()
            if line.startswith("Volume"):
               volume = ls[1]
            else:
                if len(ls)>6:
                    if journal == "pra" or journal == "prc" or journal == "pre":
                        issue = ls[1]
                        month = ls[2]
                        year = ls[3]
                        min = ls[4][1:]
                        if min.startswith("R") or min.startswith("S"):
                            min = min[1:]
                        max = ls[6][:-1]
                    else:
                        if journal == "rmp" or journal == "prx":
                            issue = ls[1]
                            month = ls[2]
                            year = ls[5]
                            min = ls[6][1:]
                            if min.startswith("S") or min.startswith("R"):
                                min = min[1:]
                            max = ls[8][:-1]
                        else:
                            issue = ls[1]
                            month = ls[3]
                            year = ls[4]
                            min = ls[5][1:]
                            if min.startswith("R") or min.startswith("S"):
                                min = min[1:]
                            max = ls[7][:-1]
                    if volumes_issues.get(journal) == None:
                        volumes_issues[journal] = {}
                    if volumes_issues.get(journal).get(volume) == None:
                        volumes_issues[journal][volume] = {}
                    if volumes_issues.get(journal).get(volume).get(month) == None:
                        volumes_issues[journal][volume][month] = {}

                    values = volumes_issues.get(journal).get(volume).get(month)
                    if values != {}:
                        values.append([min, max])
                    else:
                        volumes_issues[journal][volume][month] = [[min, max]]
#                    volumes_years[journal][volume] = year
            line = text.readline()
        text.close()
            

def doi_complete_volume(id_journal, volume, month):
    res = []
    numbers = volumes_issues[id_journal][volume][month]
    for i in numbers:
        res.append(doi_range_volume(id_journal, volume, i))
    return res


def doi_range_volume(id_journal, volume, ranges):
    res = []
    s = ranges[0]
    e = ranges[1]
    l = len(s)
    for j in range(int(s),int(e)+1):
        res.append(construct_doi(id_journal, volume, str(j).zfill(l)))
    return res


def construct_doi(id_journal, volume, number):
    values = journals_servers[id_journal]
    doi = DOI_DIR + values["abbr"] + volume + "." + number
    return doi


# def getFirstAuthor(id_journal, doi):
#     url = HOST_PHYSICAL + journals_servers[id_journal]["subdir"] + doi
#     req = requests.get(url)
#     p = req.text
#     print(p)
#     pt = "author = {"
#     start = p.index(pt) + len(pt)
#     st = p[start:]
#     author = st.split(",")[0]
#     return author


def flatten_list_dois(dois):
    flat_list = [item for sublist in dois for item in sublist]
    return flat_list

# def getFirstAuthor(id_journal, doi):
#     cu = "curl --location --header \"Accept: application/rdf+xml\""
#     url = HOST_PHYSICAL + journals_servers[id_journal]["subdir"] + doi
#     comm = cu + " " + url    
#     res = subprocess.Popen([comm],stdout=subprocess.PIPE,shell=True)
#     res.wait()
#     (t,err) = res.communicate()
#     pt = "author = {"
#     p = str(t)
#     start = p.index(pt) + len(pt)
#     st = p[start:]
#     author = st.split(",")[0]
#     return author


def downloadArticleWget(urllist, path):
    wg = "wget -e robots=off --header='Accept: application/pdf' --no-check-certificate --no-clobber --timeout=1 --tries=1 --continue "
    print("-------------")
    print(path)
    lista = urllist.split(",")
    for url in lista:
        print(url)
        sp = url.split("/")
        filename = sp[-1]
        if url == lista[-1]:
            filename = filename[0:-1]
        print(filename)
        outfile = path + "/" + filename
        print(outfile)
        exists = os.path.isfile(outfile)
        if (not exists):
            comm = wg + " -O " + os.path.join(path, filename) + " " + url
            #            print(comm)
            #            res = subprocess.Popen([comm],stdout=subprocess.PIPE,shell=True)
            #            res.wait()
            #            res = subprocess.Popen(["wget", "-e", "robots=off", "--header=\'Accept: application/pdf\'", "--no-check-certificate", "--no-clobber", "--timeout=1", "--tries=1", "--continue", "-P",path,url], stdout=subprocess.PIPE)
            res = subprocess.Popen(["wget", "-e", "robots=off", "--header=\'Accept: application/pdf\'", "--no-check-certificate", "--no-clobber", "--timeout=1", "--tries=1", "--continue", "-O",os.path.join(path, filename),url], stdout=subprocess.PIPE)
            res.wait()
            #            t = res.communicate()                                                                                                                                                   
        if (os.path.getsize(outfile)==0):
            os.remove(outfile)
        else:
            print("File Exists Not retrieving")
            break

            
# def downloadArticle(id_journal, doi):
#     wg = "wget --no-check-certificate "
#     cifrado = doi2base64(doi)
#     author = getFirstAuthor(id_journal, doi)
#     year = id_journal["year"]
#     scihub = id_journal["scihub"]
#     link = scihub + cifrado + "/" + author + year + ".pdf"
#     comm = wg + link
#     res = subprocess.Popen([comm],stdout=subprocess.PIPE,shell=True
#     res.wait()
#     t = res.communicate()[0]
#     return t


# def doi2base64(doi):
#     enc = 'utf-8'
#     t = base64.b64encode(doi.encode(enc))
#     return str(t, enc)

def downloadJSON(doi):
#    host = "http://harvest.aps.org/v2/journals/articles/"
    host = "https://sci-hub.tw/mirrors/"
    url = host + doi
    print(url)
    while True:
        try:
            filename = urlopen(url)
            print(filename)
            data = filename.read().decode('utf-8')
            json = data.replace("\\", "")
            d = ast.literal_eval(json)
            print(d)
            break
        except:
            print("Except Open URL")
            time.sleep(1)
            continue
    url = ""
    v = d[doi]
    if v != []:
        url = v
#        for i in v:
#            if not "dabamirror" in i:
#                url = i
#                break
    return (doi, url)


#def downloadArticle(url, path):
#    if url != "":
#        filename = wget.download(url, path)
#        print("\t" + filename)
#    time.sleep(1)

    
def doi2txt(dois, path):
    filename = open(path, 'w')
    for doi in dois:
        filename.write(doi+'\n')
    filename.close()
    
    
# def downloadArticle(id_journal, doi):
#     cifrado = doi2base64(doi)
#     author = getFirstAuthor(id_journal, doi)
#     year = journals_servers[id_journal]["year"]
#     scihub = journals_servers[id_journal]["scihub"]
#     name = author + year + ".pdf"
#     link = scihub + "mirrors/" + cifrado + "/" + name
#     print(link)
#     filename = wget.download(link)
#     print(filename)
#     time.sleep(1)
#     return True



#def convert2PDF(pdf_path, article, save_path):
#    comm = "pstotext " + pdf_path + article
#    res = subprocess.Popen([comm], stdout=subprocess.PIPE, shell=True)
#    res.wait()
#    text = res.communicate()[0]
#    filename = save_path + article.replace("pdf", "txt")
#    out = open(filename, 'wb')
#    out.write(text)
#    out.close()

    
# def convert2PDF(pdf_path, article, save_path):
#     filename = save_path + article.replace("pdf", "txt")
#     comm = "pstotext " + pdf_path + article
#     with subprocess.Popen([comm], stdout=subprocess.PIPE, shell=True) as out:
#         f = open(filename, 'wb')
#         f.write(out)
#         f.close()



    
#journal = "A"
#volume = "95"
#v = doi_complete_volume(journal, volume)
#print(v)
#doi = construct_doi(journal, volume, "032323")
#print(doi)
#"10.1103/PhysRevA.96.032323"
# "10.1103/PhysRevLett.120.031104"
#author = getAuthor(journal, doi)
#print(author)
#d = doi2base64("10.1103/PhysRevA.95.032323")
#d = doi2base64("10.1103/PhysRevLett.120.031104")
#print(d)
#downloadArticle(journal, doi)

def downloadPDF(file):
    rt = file.split("/")
    sp = rt[-1].split("_")
    journal = sp[0]
    volume = sp[1]
    month = sp[2]
    year = sp[3]
    year = year[0:-4]
    outpath = "/home/fer/physics/pdf_files/" + journal + "/" + volume + "/" + month + "/" + year
    #            outpath = "/tmp/pdf_files/" + journal + "/" + volume + "/" + month
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filetext = open(file, "r")
    lines = filetext.readlines()
    for line in lines:
        downloadArticleWget(line, outpath)
        #                downloadArticle(line, outpath)



                                                                                                                                            
def retrieveURL(journal):
    if not os.path.exists(path):
        os.makedirs(path)
        
#for journal in journals:
    volumes = volumes_issues[journal].keys()
    for volume in volumes:
        months = volumes_issues[journal][volume].keys()
        for month in months:
#            year = volumes_years[journal][volume]
#            outfile = journal+"_"+volume+"_"+month+"_"+year+".txt"
            outfile = journal+"_"+volume+"_"+month+".txt"
            exists = os.path.isfile(path + outfile)
            if not exists:                 
                dois = doi_complete_volume(journal, volume, month)
                urls = []
                successDois = []
                for doi in flatten_list_dois(dois):
                    print(doi)
                    r = downloadJSON(doi)
                    key, json = r
                    if json:
                        urls.append(','.join(json))
                        successDois.append(key)
                doi2txt(urls, path + outfile)
                doi2txt(successDois, succDois + outfile)


dic_volumes("/tmp/issues/")

path = "dois_files/"
succDois = "successFullDois/"

print("----------------")
print("GETTING DOIS")

from multiprocessing import Process, cpu_count
N = cpu_count()
total = []
# -------- Usando threads -------
#for i in journals:
#        t = threading.Thread(target=retrieveURL, args=(i,))
#            total.append(t)
#                t.start()
journals = "pra"
retrieveURL(journals)


#### Download the Articles PDF
print("----------------")
print("GETTING PDF")
import glob
import socket
socket.setdefaulttimeout(2)
total = []
dir = "/home/fer/physics/dois_files/"
files = glob.glob(dir+'pra*')
#for file in files:
#        downloadPDF(file)


        
#for journal in journals:
#    volumes = volumes_issues[journal]["volume"].keys()
#    for volume in volumes:
#        path = pdf_files + journal + "/" + volume
#        if not os.path.exists(path):
#            os.makedirs(path)
#        dois = doi_complete_volume(journal, volume)
#        for doi in flatten_list_dois(dois):
#            downloadArticle(downloadJSON(doi), path)

#for journal in os.listdir(pdf_files):
#    for volume in os.listdir(pdf_files + journal):
#        path = journal + "/" + volume + "/"
#        save_path = "pdf_texts/" + path
#        if not os.path.exists(save_path):
#            os.makedirs(save_path)
#        print(path)
#        for article in os.listdir(pdf_files + path):
#            print(article)
#            convert2PDF(pdf_files + path, article, save_path)
