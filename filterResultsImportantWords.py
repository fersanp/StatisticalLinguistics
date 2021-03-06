import pandas as pd
import numpy as np
import os
from getTop import getImportantWords, writeTopFile, merge_results

lapse = ["1months", "3months", "6months", "9months", "12months", "24months", "36months", "48months", "60months"]
aps = ["pra", "prb", "prc", "prd", "pre", "prl", "rmp", "prx"]

path = "results/"
outpath = "results_important/"


for i in range(1,7):
    for journal in os.listdir(path):
        for timelapse in os.listdir(path+journal+"/"):
            for gram in os.listdir(path+journal+"/"+timelapse+"/"):
                for arch in os.listdir(path+journal+"/"+timelapse+"/"+gram+"/"):
                    cpath = journal+"/"+timelapse+"/"+gram+"/"
                    print(cpath)
                    print("---------------")
                    out = outpath+cpath
                    if not os.path.exists(out):
                        os.makedirs(out)
                    wfile = out+arch
                    exists = os.path.isfile(wfile)
                    if (not exists):
                        rfile = open(path+cpath+arch, 'r').read().splitlines()
                        dic = {}
                        for i in rfile:
                                l = i.split("\t")
                                dic[l[0]]=l[1]
                        print(path+cpath+arch)
                        top = len(rfile)
                        r = getImportantWords(top, dic, "functional_words")
                        if r:
                                writeTopFile(wfile, r)
                                print("Filename " + wfile)

