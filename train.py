import nltk, sys, os, cPickle, time
from nltk.classify.naivebayes import NaiveBayesClassifier
import xml.dom.minidom


def read_format(infile,outfile):
    myin = open(infile)
    fo = open(outfile, "w")
    line = myin.readline()
    mm=""
    while(line):
        if (line[0].isalpha() or line[0].isdigit()):
            mm += line.strip('\n')
        elif (line[0] == '<' and line[1] != '/'):
            fo.write(mm)
            fo.write('\n')
            mm=""
        line=myin.readline()
    fo.write(mm)
    fo.write('\n')
    myin.close()
    fo.close()


def negate_sequence(text):
    # Detects negations and transforms negated words into "not_" form.
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    prev = None
    pprev = None
    for word in words:
        stripped = word.strip(delims)
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if prev:
            bigram = prev + " " + negated
            result.append(bigram)
            if pprev:
                trigram = pprev + " " + bigram
                result.append(trigram)
            pprev = prev
        prev = negated

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation

        if any(c in word for c in delims):
            negation = False

    return result


def cur_file_dir():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)
    

def get_words(datas):
    all_words = []
    for (words, sentiment) in datas:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def read_datas(fname, t_type):
    datas = []
    f = open(fname, 'r')
    line = f.readline()
    while line != '':
        datas.append([line, t_type])
        line = f.readline()
    f.close()
    return datas


def extract_features(document):
    document_words = set(document)
    features = {}
    
    try:
        word_features
    except:
        word_features = cPickle.load(open("WORD_FEATURES.pickle"))
        
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features


def classify_data(data):
    try:
        classifier
    except:
        classifier = cPickle.load(open("CLASSIFIER.pickle"))
        
    return \
        classifier.classify(extract_features(\
            nltk.word_tokenize("".join(negate_sequence(data)))))


def adv_enhance(text):
    adv = 0
    delims = "?.,!:;"
    result = [e.lower() for e in negate_sequence(text) if (len(e) >= 3 and e != "so")]
    for word in words:
        stripped = word.strip(delims)
        if adv != 0:
            for i in range(adj):
                result.append(word)
            adv = 0

        if any(adverb in word for adverb in ["very", "extremely", "so"\
                                                   "too", "really", "definietely"\
                                                   , "obsolutely"]):
            adv = adv + 1

        if any(c in word for c in delims):
            adv = 0

    return "".join(result)


retrain = False
if (retrain or (not os.path.isfile("CLASSIFIER.pickle")) or\
     (not os.path.isfile("WORD_FEATURES.pickle"))):
    # read in postive and negative training datas
    """
    read_format("pos_train_real.txt","pos_input.txt")
    read_format("neg_train_real.txt","neg_input.txt")
    """
    
    pos_train = read_datas('pos_train.txt', 'positive')
    neg_train = read_datas('neg_train.txt', 'negative')

    # filter away words that are less than 3 letters to form the training datas
    train = []
    for (words, sentiment) in pos_train + neg_train:
        words_filtered = [e.lower() for e in negate_sequence(\
            adv_enhance(words)) if len(e) >= 3]
        train.append((words_filtered, sentiment))


    # extract the word features out from the training datas
    word_features = get_word_features(\
                        get_words(train))
    cPickle.dump(word_features,open("WORD_FEATURES.pickle",'w'))


    # get the training set and train the Naive Bayes Classifier
    training_set = nltk.classify.util.apply_features(extract_features, train)
    print("Training the model...")
    start = time.clock()
    classifier = NaiveBayesClassifier.train(training_set)
    end = time.clock()
    print("Model training completed in %fs!" % (end-start))


    # save the model
    os.chdir(cur_file_dir())
    cPickle.dump(classifier,open("CLASSIFIER.pickle",'w'))

