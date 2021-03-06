from bs4 import BeautifulSoup
from urllib.request import urlopen
import os

HOST_PHYSICAL = "https://journals.aps.org/"
#journals = ["pra", "prb", "prc", "prd", "pre", "prl", "rmp", "prx"]
journals = ["pra", "prb", "prc", "prd", "pre"]


def getIssue(id_journal, volume):
    l = []
    url = HOST_PHYSICAL + id_journal + "/issues/" + str(volume) + "#v" + str(volume)
    print(url)
    req = urlopen(url)
    html = req.read()
    parsed_html = BeautifulSoup(html, "html5lib")
    li = parsed_html.body.findAll("li")
    i = 1
    for k in li:
        t = "/" + id_journal + "/issues/" + str(volume) + "/" + str(i)
        b = k.findAll("b")
        for j in b:
            links = j.findAll("a", href=True)
            for link in links:
                if link.attrs['href'] == t:
                    print(k.text)
                    l.append(k.text)
                    i += 1
    return l



path = "issues/"
if not os.path.exists(path):
    os.makedirs(path)


for journal in journals:
    file = open(path+journal+".txt", 'w')
    url = HOST_PHYSICAL + journal + "/issues/" 
    req = urlopen(url)
    html = req.read()
    parsed_html = BeautifulSoup(html, "html5lib")
    div = parsed_html.body.findAll("div", {"class":"volume-issue-list"})
    for i in div:
        volume = i.text.split()[1]
        file.write("Volume " + volume + "\n")
        g = getIssue(journal, volume)
        for k in g:
            file.write(k+"\n")
    file.close()
