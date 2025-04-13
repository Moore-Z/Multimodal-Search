import glob, os

def readfiles(path):
   os.chdir(path)
   pdfs = []
   for file in glob.glob("*.pdf"):
       print(file)
       pdfs.append(file)


path = "/Users/henry/Documents/Project/PracticeProject/Multimodal-Search/data/PDF"
readfiles(path)
