# Separate pdf_texts of prx and rmp by months
import os
import shutil
import re
from pathlib import Path

journals = ["rmp"]
path = "pdf_texts/"

pattern1 = "(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s\d{4}"

pattern2 = "ublished\s[0-9]+\s(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEM\
BER)\s\d{4}"

res = ""
for journal in journals:
    for volume in os.listdir(path+journal+"/"):
        filename = path+journal+"/"+volume+"/"
        for month in os.listdir(filename):
            for fil in os.listdir(filename+month):
                arch = open(filename+month+"/"+fil,'r').readlines()
                find = False
                for line in arch:
                    x = re.search(pattern2, line)
                    if x:
                        find = True
                        break
                if not find:
                    arch = open(filename+month+"/"+fil,'r').readlines()
                    for line in arch:
                        x = re.search(pattern1, line)
                        if x:
                            find = True
                            break

                if not find :
                    res = month
                else:
                    line = x.group(0)
                    st = line.split()
                    res = st[-2]

                current = filename+month+"/"+fil
                newPath = filename+res.title()+"/"
                new = newPath+fil
                if not os.path.exists(newPath):
                    print("Creating path "+ newPath)
                    os.mkdir(newPath)
                if current != new:
                    print("Moving "+ current + " " + new)    
                    shutil.move(current, new)
