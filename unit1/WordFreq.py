import string
import sys

import ijson
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

NUM_ARGS = 1


# get list of arguments besides the program name and return them in a list
def parseargs():
    args = sys.argv[1:]
    if len(args) != NUM_ARGS:
        print('Incorrect number of args', file=sys.stderr)
        exit(1)
    return args


# from the given json object, convert into sentences and from there cleaned into a list of words
def preprocessforfreq(jsonarr):
    # remove entries with empty sentences
    jsonarr = filter(lambda data: data['Sentences_t'], jsonarr)
    # change the array to just be a list of all the sentences and make them all lowercase
    sentences = map(lambda data: data['Sentences_t'].lower(), jsonarr)
    # punctuation and stopwords to remove
    stoplist = stopwords.words('english') + list(string.punctuation)
    cleanedsentences = []

    stemmer = SnowballStemmer('english')

    for sentence in sentences:
        # tokenize by words
        words = word_tokenize(sentence)
        # filter list of words to remove stop words and punctuation
        filtered = list(filter(lambda word: word not in stoplist, words))
        # get stem of all words
        stemmed = map(lambda word: stemmer.stem(word), filtered)
        cleanedsentences.extend(stemmed)
    return cleanedsentences


# TODO uncomment below to take json path as arg (delete line below that)
# jsonFilePath = parseargs()[0]
jsonFilePath = "./testDataFixed.json"

# open json file specified by path and store it as an ijson list
jsonFile = open(jsonFilePath)
jsonArr = ijson.items(jsonFile, 'item')

# preprocess the array
processed = preprocessforfreq(jsonArr)

freqDist = FreqDist(processed)
print(freqDist.most_common(10))
