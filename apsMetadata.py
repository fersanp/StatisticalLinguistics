import requests
import os
import subprocess

def downloadMetadata(doi, path):
    wg = "curl -D - -H \'Accept: application/vnd.tesseract.article+json\'"
    print("-------------")
    print(path)
    print(doi)
    doi = doi[0:-1]
    url = "http://harvest.aps.org/v2/journals/articles/" + doi
    print(url)
    filename = doi.split("/")[-1]
    outfile = path + "/" + filename
    print(outfile)
    exists = os.path.isfile(outfile)
    if (not exists):
        print(" ".join(["curl", "-D -", "-H \'Accept: vnd.tesseract.article+json\'",url, "-o",outfile]))
        res = subprocess.Popen(["curl", "-D -", "-H \'Accept: vnd.tesseract.article+json\'",url, "-o",outfile], stdout=subprocess.PIPE)
        res.wait()


def retrieveMetadata(file):
    print(file)
    rt = file.split("/")
    sp = rt[-1].split("_")
    journal = sp[0]
    volume = sp[1]
    month = sp[2]
    month = month[0:-4]
    outpath = "/home/fer/aps/metadata/" + journal + "/" + volume + "/" + month
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filetext = open(file, "r")
    lines = filetext.readlines()
    for line in lines:
        downloadMetadata(line, outpath)


pathDois = "/home/fer/aps/successFullDois/prx*"
#### Download the Metadata of Articles
print("----------------")
print("GETTING METADATA")
import glob
import socket
socket.setdefaulttimeout(2)
total = []
files = glob.glob(pathDois)
for file in files:
    retrieveMetadata(file)
    

