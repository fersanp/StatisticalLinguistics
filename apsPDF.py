import requests
import os
import subprocess

def downloadPDF(doi, path):
    wg = "curl -D - -H \'Accept: application/pdf\'"
    print("-------------")
    print(path)
    print(doi)
    doi = doi[0:-1]
    url = "http://harvest.aps.org/v2/journals/articles/" + doi
    print(url)
    filename = doi.split("/")[-1]
    outfile = path + "/" + filename + ".pdf"
    print(outfile)
    exists = os.path.isfile(outfile)
    comm = wg + " " + url + " -o " + outfile
    if (not exists):
        print(" ".join(["curl", "-D -", "-H \'Accept: application/pdf\'",url, "-o",outfile]))
        res = subprocess.Popen("curl -D - -H \'Accept: application/pdf\' " + url + " -o " + outfile, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#        res = subprocess.Popen(["curl","-D -","-H \'Accept: application/pdf\'",url,"-o",outfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        res.wait()
    return res

def retrievePDF(file):
    print(file)
    rt = file.split("/")
    sp = rt[-1].split("_")
    journal = sp[0]
    volume = sp[1]
    month = sp[2]
    month = month[0:-4]
    outpath = "/home/fer/aps/pdf_files/" + journal + "/" + volume + "/" + month
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filetext = open(file, "r")
    lines = filetext.readlines()
    for line in lines:
        downloadPDF(line, outpath)


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
            line = text.readline()
        text.close()

            
def doi_complete_volume(journal, volume, month):
    res = []
    numbers = volumes_issues[journal][volume][month]
    for i in numbers:
        res.append(doi_range_volume(journal, volume, i))
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


def flatten_list_dois(dois):
    flat_list = [item for sublist in dois for item in sublist]
    return flat_list

def retrieveURL(journal):
    volumes = volumes_issues[journal].keys()
    for volume in volumes:
        months = volumes_issues[journal][volume].keys()
        for month in months:
            outPDFpath = "/home/fer/aps/pdf_files/" + journal + "/" + volume + "/" + month
            if not os.path.exists(outPDFpath):
                os.makedirs(outPDFpath)

            outfile = journal+"_"+volume+"_"+month+".txt"
            exists = os.path.isfile(path + outfile)
            if not exists:
                dois = doi_complete_volume(journal, volume, month)
                successDois = []
                for doi in flatten_list_dois(dois):
                    print(doi)
                    r, err = downloadPDF(doi, outPDFpath)
                    print(r)
                    print(err)
                    key, json = r
                    print(key)
                    print(json)
                    if json:
                        successDois.append(doi)
#                        doi2txt(urls, path + outfile)
#                        doi2txt(successDois, succDois + outfile)



                        
pathSuccessfullDois = "/home/fer/aps/successFullDois/"
pathIssues = "/home/fer/aps/issues/"
path = "/tmp/issues/"
succDois = "successFullDois/"
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


### Create Volume Issues
#prb.txt
#volume_issues = {'prb': {'91': {'March': [['094101', '099907']],
#                                'January': [['014101', '019903'], ['020101', '024514'], ['035101', '039910'], ['041101', '045442']],
#                                'February': [['054101', '059903'], ['060101', '064513'], ['075101', '079902'], ['081101', '085433']]},
#                         '70': {'September': [['092101', '099904']],
#                                'August': [['052101', '059904'], ['060101', '069903'], ['073101', '079901'], ['081101', '087401']]}}}
volumes_issues = {}
dic_volumes("/tmp/issues/")


#### Download PDF of Articles
print("----------------")
print("GETTING PDF")
retrieveURL("pra")
import glob
import socket
socket.setdefaulttimeout(2)
total = []
files = glob.glob(pathDois+'pra*')
#print(files)
#for file in files:
#    print(file)
#    retrievePDF(file)

