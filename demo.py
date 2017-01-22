import os, nltk, cPickle
from train import cur_file_dir, extract_features, classify_data


os.chdir(cur_file_dir())

if (not os.path.isfile("CLASSIFIER.pickle")) or \
   (not os.path.isfile("WORD_FEATURES.pickle")):
    print("Please train the model first..")
    exit()

classifier = cPickle.load(open("CLASSIFIER.pickle"))
word_features = cPickle.load(open("WORD_FEATURES.pickle"))

while(1):
    data = raw_input("input the sentence you want to analyze(-1 to exit): ")
    if data == "-1":
        print("Program exitted..")
        exit()
    print(classify_data(data))
