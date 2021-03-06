import os

dir = 'pdf_files/'
for journal in os.listdir(dir):
#    journal = "prc"
    print(journal)
    for volume in os.listdir(dir+journal):
        for month in os.listdir(dir+journal+"/"+volume):
            for day in os.listdir(dir+journal+"/"+volume+"/"+month):
                fname = dir+journal+'/'+volume+'/'+month+"/"+day
#                print(fname)
                if fname.endswith("\n"):
#                    os.remove(fname)
                    print("##############")
                    print("File "+fname+ " Removed!")
                    print("##############")
                else:
                    if os.path.getsize(fname)==0:
#                        os.remove(fname)
                        print("##############")
                        print("File "+fname+ " Removed!")
                        print("File size 0")
                        print("##############")
                    else:
                        try:
                            f = open(fname, 'r')
                            line = f.readline()
#                    print(line)
                            if "not authorized" in line:
                                os.remove(fname)
                                print("##############")
                                print("File "+fname+ " Removed!")
                                print("##############")
                        except:
                            continue
                    

                   
