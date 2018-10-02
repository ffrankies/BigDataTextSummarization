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
from operator import add

import pyspark
from pyspark.sql import Row
import pyspark.sql.functions as F

spark_context = pyspark.SparkContext.getOrCreate()
spark_context.setLogLevel("OFF")
sql_context = pyspark.SQLContext(spark_context)

from nltk.corpus import wordnet
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


def rdd_show(rdd, heading="=====RDD====="):
    """data_frame.show() for an RDD

    Params:
    - rdd (pyspark.rdd.RDD): The RDD whose contents to show
    - heading (str): The heading to print above the rdd
    """
    print(heading)
    for row in rdd.take(5):
        print(row)
# End of show()


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
    - records (pyspark.rdd.RDD): The contents of the JSON file
    """
    data_frame = sql_context.read.json(file)
    data_frame.show(n=5, truncate=100)
    records = data_frame.select(constants.TEXT)
    records = records.filter(F.length(F.col(constants.TEXT)) > 99).rdd
    rdd_show(records, "=====Loaded Records=====")
    return records
# End of load_records()


def preprocess_records(records):
    """Preprocesses the records into lemma lists. Filters out any stopwords or non-dictionary words in the list.

    Params:
    - records (pyspark.sql.dataframe.DataFrame): The non-empty records from the JSON file

    Returns:
    - records_tokenized (list<list<str>>): The tokenized text content of the records
    """
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()

    records_tokenized = records.map(lambda record: tokenize_record(record, word_tokenize))
    records_tagged = records_tokenized.map(pos_tag)
    records_filtered = records_tagged.map(filter_stopwords)
    records_lemmatized = records_filtered.map(lambda record: lemmatize_record(record, lemmatizer))
    rdd_show(records_lemmatized, "=====Lemmatized Records=====")

    return records_lemmatized
# End of tokenize_records()


def tokenize_record(record, f_tokenize):
    """Tokenizes a single record (row) in the RDD.

    Params:
    - record (str): The record to be tokenized
    - f_tokenize (function): The function doing the tokenization
    """
    lowercase = record[constants.TEXT].encode('utf-8').lower()
    tokenized = f_tokenize(lowercase)
    return tokenized
# End of tokenize_record()


def filter_stopwords(tagged_record):
    """Filters stopwords, punctuation, and contractions from the tagged records. This is done after tagging to make
    sure that the tagging is as accurate as possible.

    Params:
    - tagged_record (list<tuple<str, str>>): The records, with each word tagged with its part of speech

    Returns:
    - filtered_record (list<tuple<str, str>>): The records, with unimportant words filtered out
    """
    print('type of row = ', type(tagged_record))
    from nltk.corpus import stopwords
    from nltk.corpus import words as nltk_words

    stop_words = list(stopwords.words('english'))
    stop_words.extend(string.punctuation)
    stop_words.extend(constants.CONTRACTIONS)
    stop_words.extend(constants.MYSQL_STOPWORDS)
    dictionary_words = set(nltk_words.words())

    def not_dictionary_word(word): 
        return word[0] not in dictionary_words and word[1] not in ['NNP', 'NNPS']

    filtered_record = filter(lambda word: word[0] not in stop_words, tagged_record)
    filtered_record = filter(not_dictionary_word, filtered_record)
    filtered_record = filter(lambda word: not word[0].replace('.', '', 1).isdigit(), filtered_record)
    filtered_record = list(filter(lambda word: word[1] in POS_TRANSLATOR.keys(), filtered_record))
    return filtered_record
# End of filter_stopwords()


def lemmatize_record(tagged_record, lemma_model):
    """Lemmatizes the words in the given record, provided they're tagged with their correct part of speech.

    Params:
    - tagged_record (???): The record, as a list of tokens with their part of speech tag
    - lemma_model (wordnet.Lemmatizer): The lemmatizer model

    Returns:
    - record_lemmatized (list<str, str>): The tagged lemmas for each word in the record
    """
    record_lemmatized = [lemma_model.lemmatize(word[0], POS_TRANSLATOR[word[1]]).encode('utf-8')
                         for word in tagged_record]
    return record_lemmatized
# End of lemmatize_record()


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
    # tagged_records = map(lambda record: pos_tag(record), records)
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
    # @see https://github.com/apache/spark/blob/master/examples/src/main/python/wordcount.py
    word_counts = records.flatMap(lambda x: x)\
        .map(lambda x: (x, 1))\
        .reduceByKey(add)\
        .sortBy(lambda a: a[1], ascending=False)
    rdd_show(word_counts, "=====Word Counts=====")

    frequent_words = word_counts.take(num_words)
    if no_counts:
        frequent_words = [word[0] for word in frequent_words]
    print("=====The {:d} Most Frequent Words=====".format(num_words))
    print(frequent_words)
    return frequent_words
# End of extract_frequent_words()


def extract_collocations(records, num_collocations, collocation_window, compare_collocations=False):
    """Extracts the most common collocations present in the records.

    Params:
    - records (list<list<str>>): The tokenized and lemmatized records from the JSON file
    - num_collocations (int): The number of collocations to show
    - collocation_window (int): The text window within which to search for collocations

    Returns:
    - best_collocations (list<tuple<str>>): The highest scored collocations present in the records
    """
    # @see: https://spark.apache.org/docs/2.2.0/ml-features.html#n-gram
    from pyspark.ml.feature import NGram
    
    data_frame = records.map(lambda l: Row(l)).toDF(['words'])
    ngram_model = NGram(n=2, inputCol='words', outputCol='ngrams')
    ngram_data_frame = ngram_model.transform(data_frame)
    
    ngram_rdd = ngram_data_frame.select('ngrams').rdd
    ngram_rdd = ngram_rdd.flatMap(lambda row: row['ngrams'])\
        .map(lambda ngram: (ngram.encode('utf-8'), 1))\
        .reduceByKey(add)\
        .sortBy(lambda x: x[1], ascending=False)
    rdd_show(ngram_rdd)

    frequent_collocations = ngram_rdd.take(num_collocations)
    print("=====The {:d} Most Frequent Collocations=====".format(num_collocations))
    pprint.pprint(frequent_collocations)

    return frequent_collocations
# End of extract_collocations()


if __name__ == "__main__":
    args = parse_arguments()
    records = load_records(args.file)
    records_lemmatized = preprocess_records(records)
    extract_frequent_words(records_lemmatized, args.num_words, False)
    extract_collocations(records_lemmatized, args.num_collocations, args.collocation_window, False)
