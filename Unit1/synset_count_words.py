# Unit 1, read json file, and change it to the wanted format

import json
import re
import string
import data_area
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


def get_records_from_file(file_name):
    '''
    get records from the json file, save them in list
    :param file_name: file (str): The path to the JSON file
    :return: the list of json records
    '''
    with open(file_name) as input_f:
        lines = input_f.readlines()

    records = []
    for record in lines:
        records.append(json.loads(record))

    return records


def remove_stopwords(input_data):
    '''
    remove all the stop words
    :param input_data: list of words
    :return: list of words without stop words
    '''

    filtered_words = []
    stoplist = stopwords.words('english') + list(string.punctuation) \
               + data_area.MYSQL_STOPWORDS
    for word in input_data:
        if word[0] not in stoplist:
            filtered_words.append(word)

    return filtered_words


def get_original_words(input_data):
    '''
    get words for the json records' Sentences_t part, change them to lower case, and divide them to a words list
    :param input_data: the json record list
    :return: a list of words that has all the word in the list and in lower case
    '''
    word_list = []
    for line in input_data:
        words_lower = (line['Sentences_t'].lower())
        for word in re.findall(r'\w+', words_lower):
            word_list.append(word)
    return word_list


def get_root_words_list(input_data):
    '''
    get the all the root of words from the json records' Sentences_t part as a list
    :param input_data: the json record list
    :return: a list of root of words that in lower case
    '''
    word_list = []
    word_roots = nltk.SnowballStemmer("english")
    for line in input_data:
        words_lower = (line['Sentences_t'].lower())
        for word in re.findall(r'\w+', words_lower):
            word_list.append(word_roots.stem(word))
    return word_list



def change_to_present(input_list):
    '''
    change all the words in the input list to present tense
    :param input_list: a list of words
    :return: same list of words, but all the words are in present tense
    '''
    present_tense_words_list = []
    for word in input_list:
        present_tense_words_list.append(nltk.WordNetLemmatizer().lemmatize(word, 'v'))
    return present_tense_words_list


def get_syn_list(word):
    '''
    take a give word, find its synset
    :param word: the target word
    :return: a set of synonyms of the target word
    '''
    sylist = []
    t_set = wn.synsets(word)
    for synset in t_set:
        for lemma in synset.lemmas():
            sylist.append(lemma.name())
    return sylist


def get_synset_set(input_list):
    '''
    group words which have same meaning
    :param input_list: a list of words
    :return:  a list of group words which have same meaning
    '''
    set_list = []
    for word in input_list:
        set = []
        if word not in set_list and word not in set:
            syn_words = get_syn_list(word[0])
            for iword in input_list:
                if iword[0] in syn_words:
                    set.append(iword)

        set_list.append(set)
    return set_list


def count_root_words(input_data):
    '''
    get the root of words from the json record list, and count the root of words
    :param input_data: the json record
    :return: the result
    '''
    word_list = get_root_words_list(input_data)
    clist = Counter(word_list)
    sclist = clist.most_common()
    flist = remove_stopwords(sclist)
    return flist


def count_ori_words(input_data):
    '''
    get words from json record list, and count them
    :param input_data:  the json record
    :return: the result
    '''
    word_list = get_original_words(input_data)
    clist = Counter(word_list)
    sclist = clist.most_common()
    flist = remove_stopwords(sclist)
    return flist

def get_synset_group(input_data):
    '''
    take the json records, get words and change tends, group the words who have same meaning
    :param input_data: the json records
    :return: the list of synset
    '''
    word_list = get_original_words(input_data)
    plist = change_to_present(word_list)
    sclist = Counter(plist).most_common()
    flist = remove_stopwords(sclist)
    syn_list = get_synset_set(flist)
    return syn_list



def start():
    '''
    you can choose what you want to run in there, from:
    result_list = count_ori_words(records)
    result_list = count_root_words(records)
    result_list = get_synset_group(records)
    :return: none
    '''

    records = get_records_from_file("test.json")
    # result_list = count_ori_words(records)
    # result_list = count_root_words(records)
    result_list = get_synset_group(records)

    for record in result_list:
        print (record)



start()
