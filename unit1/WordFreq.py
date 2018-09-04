import argparse
import json
import string

from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

TEXT = 'Sentences_t'
NUM_ARGS = 1


def parse_arguments():
    """Parses command-line arguments.
    Returns:
    - args (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='The path to the JSON file containing processed text')
    return parser.parse_args()
# End of parse_arguments()


def load_records(file):
    """Loads the records from the JSON file. Also filters out empty records.
    Params:
    - file (str): The path to the JSON file
    Returns:
    - records (list<dict>): The contents of the JSON file
    """
    with open(file, 'r') as json_file:
        records = json_file.readlines()
    records = [json.loads(record) for record in records]
    records = list(filter(lambda record: record[TEXT] != '', records))
    return records
# End of load_records()


def preprocess_freq(jsonarr):
    """
    from the given json object, convert into sentences and from there cleaned into a list of words
    :param jsonarr: a list of json data on the articles
    :return: a list of important words
    """
    # remove entries with empty sentences
    jsonarr = filter(lambda data: data[TEXT], jsonarr)
    # change the array to just be a list of all the sentences and make them all lowercase
    sentences = map(lambda data: data[TEXT].lower(), jsonarr)
    # punctuation and stopwords to remove
    stoplist = stopwords.words('english') + list(string.punctuation)

    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')

    cleaned = []
    for sentence in sentences:
        # tokenize by words
        words = word_tokenize(sentence)
        # filter list of words to remove stop words and punctuation
        filtered = list(filter(lambda word: word not in stoplist, words))
        # get stem of all words
        stemmed = map(lambda word: stemmer.stem(word), filtered)
        # lemmatize all words
        lemmatized = map(lambda word: lemmatizer.lemmatize(word), stemmed)

        cleaned.extend(lemmatized)

    return cleaned
# End of preprocess_freq


def get_freq(wordlist, count=10):
    freqDist = FreqDist(wordlist)
    return freqDist.most_common(count)
#end of get_freq


if __name__ == "__main__":
    jsonFilePath = vars(parse_arguments())['file']
    jsonArr = load_records(jsonFilePath)
    processed = preprocess_freq(jsonArr)
    print(get_freq(processed, 50))
