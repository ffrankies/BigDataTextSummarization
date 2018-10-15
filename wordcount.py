# coding: utf-8
"""Using word frequencies to create a summary.
"""
# from sparknlp.base import *

import argparse
import string
import pprint
from operator import add

import pyspark
from pyspark.sql import Row
import pyspark.sql.functions as F

spark_context = pyspark.SparkContext.getOrCreate()
spark_context.setLogLevel("OFF")
sql_context = pyspark.SQLContext(spark_context)

from nltk.corpus import wordnet

import constants

###########################
# Set up nltk
###########################
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

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
    'WRB': wordnet.ADV     # wh-adverb  where, when
}

TEXT_FIELD = constants.TEXT
CONTRACTIONS = constants.CONTRACTIONS
MYSQL_STOPWORDS = constants.MYSQL_STOPWORDS


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
    records = data_frame.select(TEXT_FIELD)
    # Filter out records shorter than 100 characters. Then,
    # add a unique ID to each record. Then,
    # make the ID the first element in the record, so it can be used as a key
    records = records.filter(F.length(F.col(TEXT_FIELD)) > 99).rdd\
        .zipWithUniqueId()\
        .map(lambda record: (record[1], record[0]))
    rdd_show(records, "=====Loaded Records=====")
    return records
# End of load_records()


def preprocess_records(records):
    """Preprocesses the records into lemma lists. Filters out any stopwords or non-dictionary words in the list.

    Params:
    - records (pyspark.rdd.RDD): The non-empty records from the JSON file

    Returns:
    - records_lemmatized (pyspark.rdd.RDD): The tokenized text content of the records
    """
    lemmatizer = WordNetLemmatizer()

    records_tokenized = records.map(tokenize_record)
    records_tagged = records_tokenized.map(lambda record: (record[constants.KEY], pos_tag(record[constants.VALUE])))
    records_filtered = records_tagged.map(filter_stopwords)
    records_lemmatized = records_filtered.map(lambda record: lemmatize_record(record, lemmatizer))
    rdd_show(records_lemmatized, "=====Preprocessed Records=====")

    return records_lemmatized
# End of tokenize_records()


def tokenize_record(record):
    """Tokenizes a single record (row) in the RDD.

    Params:
    - record (tuple<int, dict>): The record to be tokenized, as a key-value pair

    Returns:
    - key (int): The ID of the record
    - tokenized (list<str>): The list of tokens present in the record
    """
    import nltk

    key, record_contents = record
    lowercase = record_contents[TEXT_FIELD].encode('utf-8').lower()
    tokenized = nltk.word_tokenize(lowercase)
    return (key, tokenized)
# End of tokenize_record()


def filter_stopwords(tagged_record):
    """Filters stopwords, punctuation, and contractions from the tagged records. This is done after tagging to make
    sure that the tagging is as accurate as possible.

    Params:
    - tagged_record (tuple<int, list>): The record, with each word tagged with its part of speech

    Returns:
    - key (int): The ID of the record
    - filtered_record (list<tuple<str, str>>): The POS-tagged record tokens, with stopword tokens filtered out
    """
    from nltk.corpus import stopwords
    from nltk.corpus import words as nltk_words

    key, record = tagged_record
    stop_words = list(stopwords.words('english'))
    stop_words.extend(string.punctuation)
    stop_words.extend(CONTRACTIONS)
    stop_words.extend(MYSQL_STOPWORDS)
    dictionary_words = set(nltk_words.words())

    def noun_or_dictionary_word(word):
        return word[0] in dictionary_words or word[1] in ['NNP', 'NNPS']

    filtered_record = filter(lambda word: word[0] not in stop_words, record)
    filtered_record = filter(noun_or_dictionary_word, filtered_record)
    filtered_record = filter(lambda word: not word[0].replace('.', '', 1).isdigit(), filtered_record)
    filtered_record = filter(lambda word: word[1] in POS_TRANSLATOR.keys(), filtered_record)
    return (key, list(filtered_record))
# End of filter_stopwords()


def lemmatize_record(tagged_record, lemma_model):
    """Lemmatizes the words in the given record, provided they're tagged with their correct part of speech. The part
    of speech tagged is stripped away from the words.

    Params:
    - tagged_record (tuple<int, list>): The record, as a list of tokens with their part of speech tag
    - lemma_model (wordnet.Lemmatizer): The lemmatizer model

    Returns:
    - id (int): The ID of the record
    - record_lemmatized (list<str>): The lemmas for each word in the record.
    """
    key, record = tagged_record
    record_lemmatized = [lemma_model.lemmatize(word[0], POS_TRANSLATOR[word[1]]).encode('utf-8')
                         for word in record]
    return (key, record_lemmatized)
# End of lemmatize_record()


def extract_frequent_words(records, num_words, no_counts=False):
    """Stems the words in the given records, and then counts the words using NLTK FreqDist.

    Stemming is done using the English Snowball stemmer as per the recommendation from
    http://www.nltk.org/howto/stem.html

    NB: There is also a Lancaster stemmer available, but it is apparently very aggressive and can lead to a loss of
    potentially useful words (source: https://stackoverflow.com/a/11210358/5760608)

    Params:
    - records (pyspark.rdd.RDD): The tokenized records from the JSON file
    - num_words (int): The number of words to extract
    - no_counts (bool): If True, frequent words will not include the word counts

    Returns:
    - frequent_words (list<str> or list<tuple<str, int>>): The list of most frequent words
    """
    # @see https://github.com/apache/spark/blob/master/examples/src/main/python/wordcount.py
    word_counts = records.flatMap(lambda record: record[constants.VALUE])\
        .map(lambda word: (word, 1))\
        .reduceByKey(add)\
        .sortBy(lambda word_with_count: word_with_count[1], ascending=False)
    rdd_show(word_counts, "=====Word Counts=====")

    frequent_words = word_counts.take(num_words)
    if no_counts:
        frequent_words = [word[0] for word in frequent_words]
    return frequent_words
# End of extract_frequent_words()


def extract_collocations(records, num_collocations, collocation_window):
    """Extracts the most common collocations present in the records.

    Params:
    - records (pyspark.rdd.RDD): The tokenized and lemmatized records from the JSON file
    - num_collocations (int): The number of collocations to show
    - collocation_window (int): The text window within which to search for collocations.

    Returns:
    - best_collocations (list<tuple<str, int>>): The highest scored collocations present in the records, with their
                                                 frequency of occurrence in the dataset.
    """
    # @see: https://spark.apache.org/docs/2.2.0/ml-features.html#n-gram
    from pyspark.ml.feature import NGram

    data_frame = records.map(lambda record: Row(record[constants.VALUE])).toDF(['words'])
    ngram_model = NGram(n=2, inputCol='words', outputCol='ngrams')
    ngram_data_frame = ngram_model.transform(data_frame)

    ngram_rdd = ngram_data_frame.select('ngrams').rdd
    ngram_rdd = ngram_rdd.flatMap(lambda row: row['ngrams'])\
        .map(lambda ngram: (ngram.encode('utf-8'), 1))\
        .reduceByKey(add)\
        .sortBy(lambda bigram_with_count: bigram_with_count[1], ascending=False)
    rdd_show(ngram_rdd)

    frequent_collocations = ngram_rdd.take(num_collocations)

    return frequent_collocations
# End of extract_collocations()


def merge_collocations_with_wordlist(collocations, wordlist):
    """Filters collocations to only return those where both words are in the wordlist. Then returns the collocations
    and the words from the wordlist that aren't in any collocations.

    Params:
    - collocations (list<tuple<str, int>>): The collocations to merge with the word list, with their counts
    - wordlist (list<str, int>): The words to filter by, with their counts

    Returns:
    - merged_list (list<str>): List of words and collocations
    """
    words = [word[0] for word in wordlist]
    word_pairs = [collocation[0] for collocation in collocations]
    unused_words = set(words)
    merged_list = list()
    for collocation in word_pairs:  # collocations look like: "one two"
        word_one, word_two = collocation.split(" ")
        if word_one != word_two and word_one in unused_words and word_two in unused_words:
            merged_list.append(collocation)
            unused_words -= set([word_one, word_two])
    merged_list.extend(unused_words)
    return merged_list
# End of merge_collocations_with_wordlist()


if __name__ == "__main__":
    args = parse_arguments()
    records = load_records(args.file)
    records_lemmatized = preprocess_records(records)
    frequent_words = extract_frequent_words(records_lemmatized, args.num_words, False)
    print("=====The {:d} Most Frequent Words=====".format(args.num_words))
    print(frequent_words)
    frequent_collocations = extract_collocations(records_lemmatized, args.num_collocations, args.collocation_window)
    print("=====The {:d} Most Frequent Collocations=====".format(args.num_collocations))
    pprint.pprint(frequent_collocations)
    # Strip counts
    words_and_collocations = merge_collocations_with_wordlist(frequent_collocations, frequent_words)
    print("=====Most Frequent Words And Collocations=====")
    print(words_and_collocations)
