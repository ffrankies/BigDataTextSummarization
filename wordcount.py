# coding: utf-8
"""Using word frequencies to create a summary.
"""
# from sparknlp.base import *

import io
import argparse
import json
import string
import random
import pprint

import pyspark
import pyspark.sql.functions as F

spark_context = pyspark.SparkContext.getOrCreate()
spark_context.setLogLevel("OFF")
sql_context = pyspark.SQLContext(spark_context)

from nltk import pos_tag
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

import constants


###########################
# PART OF SPEECH TAG TRANSLATOR FROM `pos_tag` TAGS to `wordnet` TAGS
###########################
# source for tags: https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
# NB: wordnet has a ADV_SAT tag, but I have no idea what that is
DEFAULT_TAG = wordnet.NOUN

POS_TRANSLATOR = {
    'CC': DEFAULT_TAG,     # coordinating conjunction
    'CD': DEFAULT_TAG,     # cardinal digit
    'DT': DEFAULT_TAG,     # determiner
    'EX': DEFAULT_TAG,     # existential there (like: "there is" ... think of it like "there exists")
    'FW': DEFAULT_TAG,     # foreign word
    'IN': DEFAULT_TAG,     # preposition/subordinating conjunction
    'JJ': wordnet.ADJ,     # adjective    'big'
    'JJR': wordnet.ADJ,    # adjective, comparative   'bigger'
    'JJS': wordnet.ADJ,    # adjective, superlative   'biggest'
    'LS': DEFAULT_TAG,     # list marker   1)
    'MD': wordnet.VERB,    # modal   could, will
    'NN': wordnet.NOUN,    # noun, singular 'desk'
    'NNS': wordnet.NOUN,   # noun plural   'desks'
    'NNP': wordnet.NOUN,   # proper noun, singular   'Harrison'
    'NNPS': wordnet.NOUN,  # proper noun, plural   'Americans'
    'PDT': wordnet.ADJ,    # predeterminer   'all the kids'
    'POS': DEFAULT_TAG,    # possessive ending   parent's
    'PRP': DEFAULT_TAG,    # personal pronoun   I, he, she
    'PRP$': DEFAULT_TAG,   # possessive pronoun   my, his, hers
    'RB': wordnet.ADV,     # adverb   very, silently,
    'RBR': wordnet.ADV,    # adverb, comparative   better
    'RBS': wordnet.ADV,    # adverb, superlative   best
    'RP': wordnet.ADV,     # particle   give up
    'TO': DEFAULT_TAG,     # to   go 'to' the store.
    'UH': DEFAULT_TAG,     # interjection   errrrrrrrm
    'VB': wordnet.VERB,    # verb, base form   take
    'VBD': wordnet.VERB,   # verb, past tense   took
    'VBG': wordnet.VERB,   # verb, gerund/present participle   taking
    'VBN': wordnet.VERB,   # verb, past participle   taken
    'VBP': wordnet.VERB,   # verb, sing. present, non-3d   take
    'VBZ': wordnet.VERB,   # verb, 3rd person sing. present   takes
    'WDT': DEFAULT_TAG,    # wh-determiner   which
    'WP': DEFAULT_TAG,     # wh-pronoun   who, what
    'WP$': DEFAULT_TAG,    # possessive wh-pronoun   whose
    'WRB': wordnet.ADV     # wh-abverb  where, when
}


def parse_arguments():
    """Parses command-line arguments.

    Returns:
    - args (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='The path to the JSON file containing processed text')
    parser.add_argument('-w', '--num_words', type=int, help='The number of frequent words to print out', default=20)
    parser.add_argument('-c', '--num_collocations', type=int, help='The number of collocations to print out',
                        default=10)
    parser.add_argument('-cw', '--collocation_window', type=int, help='The window for searching for collocations',
                        default=5)
    return parser.parse_args()
# End of parse_arguments()


def load_records(file, preview_records=False):
    """Loads the records from the JSON file. Also filters out empty records.

    Params:
    - file (str): The path to the JSON file

    Returns:
    - records (list<dict>): The contents of the JSON file
    """
    data_frame = sql_context.read.json(file)
    data_frame.show(n=5, truncate=100)
    records = data_frame.select(constants.TEXT)
    records = records.filter(F.length(F.col(constants.TEXT)) > 99)
    records.show(n=5, truncate=100)
    return records
# End of load_records()


def tokenize_records(records):
    """Tokenizes the records into word lists. Filters out any stopwords in the list.

    Params:
    - records (list<dict>): The non-empty records from the JSON file

    Returns:
    - records_tokenized (list<list<str>>): The tokenized text content of the records
    """
    contents = map(lambda record: record[constants.TEXT].encode('utf-8'), records)
    records_tokenized = [word_tokenize(record.lower()) for record in contents]
    records_lemmatized = lemmatize_records(records_tokenized)
    lemmatized_words = list()
    for record in records_lemmatized:
        lemmatized_words.extend(record)
    return lemmatized_words
# End of tokenize_records()


def lemmatize_records(records):
    """Lemmatizes the words in the tokenized sentences.

    Lemmatization works best when the words are tagged with their corresponding part of speech, so the words are first
    tagged using nltk's `pos_tag` function.

    NB: There is a good chance that this tagging isn't 100% accurate. For that matter, lemmatization isn't always 100%
    accurate.

    Params:
    - records (list<list<str>>): The word-tokenized records

    Returns:
    - records_lemmatized (list<str>)): The lemmatized words from all the records
    """
    print('Length of tagged_records: {:d}'.format(len(records)))
    print('Total number of words: {:d}'.format(sum([len(record) for record in records])))
    tagged_records = map(lambda record: pos_tag(record), records)
    tagged_records = filter_stopwords(tagged_records)
    lemmatizer = WordNetLemmatizer()
    records_lemmatized = list()
    for record in tagged_records:
        try:
            lemmatized_record = list(
                map(lambda word: lemmatizer.lemmatize(word[0], POS_TRANSLATOR[word[1]]).encode('utf-8'), record)
            )
        except Exception as err:
            print(record)
            raise err
        records_lemmatized.append(lemmatized_record)
    print('Total number of words after filtering: {:d}'.format(len(records_lemmatized)))
    return records_lemmatized
# End of lemmatize_records()


def filter_stopwords(tagged_records):
    """Filters stopwords, punctuation, and contractions from the tagged records. This is done after tagging to make
    sure that the tagging is as accurate as possible.

    Params:
    - tagged_records (list<list<tuple<str, str>>>): The records, with each word tagged with its part of speech

    Returns:
    - filtered_records (list<list<tuple<str, str>>>): The records, with unimportant words filtered out
    """
    print('Filtering stopwords')
    stop_words = list(stopwords.words('english'))
    stop_words.extend(string.punctuation)
    stop_words.extend(constants.CONTRACTIONS)
    stop_words.extend(constants.MYSQL_STOPWORDS)
    dictionary_words = set(nltk_words.words())

    def not_dictionary_word(word): 
        return word[0] not in dictionary_words and word[1] not in ['NNP', 'NNPS']

    filtered_records = [filter(lambda word: word[0] not in stop_words, record) for record in tagged_records]
    filtered_records = [filter(lambda word: not_dictionary_word, record) for record in filtered_records]
    filtered_records = [filter(lambda word: not word[0].replace('.', '', 1).isdigit(), record)
                        for record in filtered_records]  # see https://stackoverflow.com/a/23639915/5760608
    filtered_records = [list(filter(lambda word: word[1] in POS_TRANSLATOR.keys(), record))
                        for record in filtered_records]
    return filtered_records
# End of filter_stopwords()


def extract_frequent_words(records, num_words, no_counts=False):
    """Stems the words in the given records, and then counts the words using NLTK FreqDist.

    Stemming is done using the English Snowball stemmer as per the recommendation from 
    http://www.nltk.org/howto/stem.html

    NB: There is also a Lancaster stemmer available, but it is apparently very aggressive and can lead to a loss of
    potentially useful words (source: https://stackoverflow.com/a/11210358/5760608)

    Params:
    - records (list<str>): The tokenized records from the JSON file
    - num_words (int): The number of words to extract
    - no_counts (bool): If True, frequent words will not include the word counts

    Returns:
    - frequent_words (list<str> or list<tuple<str, int>>): The list of most frequent words
    """
    word_counts = FreqDist(records)
    frequent_words = word_counts.most_common(num_words)
    if no_counts:
        frequent_words = [word[0] for word in frequent_words]
    print("=====The {:d} Most Frequent Words=====".format(num_words))
    print(frequent_words)
    return frequent_words
# End of extract_frequent_words()


def extract_collocations(records, num_collocations, collocation_window, compare_collocations = False):
    """Extracts the most common collocations present in the records.

    Params:
    - records (list<list<str>>): The tokenized and lemmatized records from the JSON file
    - num_collocations (int): The number of collocations to show
    - collocation_window (int): The text window within which to search for collocations

    Returns:
    - best_collocations (list<tuple<str>>): The highest scored collocations present in the records
    """
    bigram_measures = BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(records, window_size=collocation_window)
    bigram_finder.apply_freq_filter(min_freq=3)
    best_collocations = bigram_finder.nbest(bigram_measures.raw_freq, num_collocations)
    print("=====The {:d} Most Frequent Collocations=====".format(num_collocations))
    pprint.pprint(best_collocations)
    if compare_collocations:
        print("=====The {:d} Best Collocations (Pointwise Mutual Information)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.pmi, num_collocations))
        print("=====The {:d} Best Collocations (Student's t test)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.student_t, num_collocations))
        print("=====The {:d} Best Collocations (Chi-square test)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.chi_sq, num_collocations))
        print("=====The {:d} Best Collocations (Mutual Information)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.mi_like, num_collocations))
        print("=====The {:d} Best Collocations (Likelihood Ratios)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.likelihood_ratio, num_collocations))
        print("=====The {:d} Best Collocations (Poisson Stirling)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.poisson_stirling, num_collocations))
        print("=====The {:d} Best Collocations (Jaccard Index)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.jaccard, num_collocations))
        print("=====The {:d} Best Collocations (Phi-square test)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.phi_sq, num_collocations))
        print("=====The {:d} Best Collocations (Fisher's Exact Test)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.fisher, num_collocations))
        print("=====The {:d} Best Collocations (Dice's Coefficient)=====".format(num_collocations))
        pprint.pprint(bigram_finder.nbest(bigram_measures.dice, num_collocations))
    return best_collocations
# End of extract_collocations()


if __name__ == "__main__":
    args = parse_arguments()
    records = load_records(args.file)
    records_tokenized = tokenize_records(records)
    extract_frequent_words(records_tokenized, args.num_words, True)
    extract_collocations(records_tokenized, args.num_collocations, args.collocation_window, False)
