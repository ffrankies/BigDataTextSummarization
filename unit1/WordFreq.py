import argparse
import json
import string
import constants

from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet


def parse_arguments():
    """Parses command-line arguments.
    Returns:
    - args (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='The path to the JSON file containing processed text.')
    parser.add_argument('-n', '--number', type=int, help='The numbers of words that this program will output.')
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
    records = list(filter(lambda record: record[constants.TEXT] != '', records))
    return records
# End of load_records()


def preprocess_freq(jsonarr):
    """
    from the given json object, convert into sentences and from there cleaned into a list of words
    :param jsonarr: a list of json data on the articles
    :return: a list of important words
    """
    # remove entries with empty sentences
    jsonarr = filter(lambda data: data[constants.TEXT], jsonarr)
    # change the array to just be a list of all the sentences and make them all lowercase
    sentences = map(lambda data: data[constants.TEXT].lower(), jsonarr)
    # punctuation and stopwords to remove
    stoplist = stopwords.words('english') + list(string.punctuation) \
               + constants.CONTRACTIONS + constants.MYSQL_STOPWORDS

    lemmatizer = WordNetLemmatizer()

    cleaned = []
    for sentence in sentences:
        # tokenize by words
        words = word_tokenize(sentence)
        # filter list of words to remove stop words and punctuation
        filtered = list(filter(lambda word: word not in stoplist, words))

        # lemmatize all words
        lemmatized = map(lambda word: lemmatizer.lemmatize(word), filtered)
        cleaned.extend(lemmatized)

    return cleaned
# End of preprocess_freq


def generate_syn_set(freq_list, rng=10):
    freq_hash = set(freq_list)
    general_synonyms = {}
    relevant_synonyms = {}

    # Get the synonyms and their hypernyms for each of the top 10 most frequent works and store the word mapped to them
    for i in range(rng):
        curr_general_synonyms = wordnet.synsets(freq_list[i])
        extended_synonyms = []
        for curr_syn in curr_general_synonyms:
            extended_synonyms.extend(curr_syn.hypernyms())
            extended_synonyms.append(curr_syn)

        extended_synonyms = list(map(lambda x: x.name().split('.')[0], extended_synonyms))
        # TODO remove print(freq_list[i], extended_synonyms)
        general_synonyms.update({freq_list[i]: extended_synonyms})

    # Store 10 most frequent works in relevant_synonyms mapped to an empty array
    for j in range(rng):
        relevant_synonyms.update({freq_list[j]: []})

    for word in general_synonyms:
        curr_syns = general_synonyms[word]
        for syn in curr_syns:
            if syn != word and syn in freq_hash and syn not in relevant_synonyms[word]:
                relevant_synonyms[word].append(syn)

    return relevant_synonyms
# End of generate_syn_set


def path_similarity_set(freq_list, rng=10):
    similarity_dict = {}
    for i in range(rng):
        similar_arr = []
        for word in freq_list:
            freq_word = freq_list[i]
            if word != freq_word and is_similar(freq_word, word):
                similar_arr.append(word)
        similarity_dict[freq_word] = similar_arr
    return similarity_dict
# End of path_similarity_set


def is_similar(word1, word2):
    set1 = wordnet.synsets(word1)
    set2 = wordnet.synsets(word2)

    for word1 in set1:
        for word2 in set2:
            score = wordnet.path_similarity(word1, word2)
            if score and score > constants.SIMILARITY_THRESHOLD:
                return True
    return False
# End of is_similar


def get_freq(word_list, count=10):
    freq_dist = FreqDist(word_list)
    return freq_dist.most_common(count)
# End of get_freq


if __name__ == "__main__":
    args = parse_arguments()
    jsonFilePath = args.file
    num = args.number

    jsonArr = load_records(jsonFilePath)
    processed = preprocess_freq(jsonArr)

    freq_words = get_freq(processed, 1000)
    no_count_freq_words = list(map(lambda tup: tup[0], freq_words))

    # TODO print(freq_words, '\n')

    syns = generate_syn_set(no_count_freq_words)
    syns2 = path_similarity_set(no_count_freq_words)

    for s in syns:
        print('%10s: ' % s, syns[s])

    print()

    for s in syns2:
        print('%10s: ' % s, syns2[s])
# End main
