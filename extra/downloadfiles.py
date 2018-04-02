import requests
import time

start_time = time.time()
print("Downloading files for training")
i=0
with open("train.txt",'r') as trainfile:
    for line in trainfile:
        line=line.rstrip()
        print("Processing "+line+" as cilia"+str(i)+".tar")
        url = "https://storage.googleapis.com/uga-dsp/project4/data/"+line+".tar"
        r = requests.get(url)
        with open("train/cilia"+str(i)+".tar", 'wb') as file:
            file.write(r.content)
        i+=1
print("Downloading masks")
i=0
with open("train.txt",'r') as trainfile:
    for line in trainfile:
        line=line.rstrip()
        print("Processing "+line+" as cilia"+str(i)+".png")
        url = "https://storage.googleapis.com/uga-dsp/project4/masks/"+line+".png"
        r = requests.get(url)
        with open("train/mask/cilia"+str(i)+".png", 'wb') as file:
            file.write(r.content)
        i+=1
print("Downloading files for testing")
i=0
with open("test.txt",'r') as testfile:
    for line in testfile:
        line=line.rstrip()
        print("Processing "+line+" as cilia"+str(i)+".tar")
        url = "https://storage.googleapis.com/uga-dsp/project4/data/"+line+".tar"
        r = requests.get(url)
        with open("test/cilia"+str(i)+".tar", 'wb') as file:
            file.write(r.content)
        i+=1
print("--- %s seconds ---" % (time.time() - start_time))
