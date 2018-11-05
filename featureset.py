# coding: utf-8
"""Constructs bag-of-words-inspired feature sets out of the preprocessed records.
"""
import json
import argparse

import pyspark.sql.functions as F
from pyspark.sql.types import Row
from nltk.tokenize import sent_tokenize

import wordcount
import tfidf
import synsets
import constants


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
    parser.add_argument('-s', '--sentences', type=bool, action='store_true')
    parser.add_argument('-o', '--output', type=str, help='The directory in which to save the featureset dataset')
    return parser.parse_args()
# End of parse_arguments()


def load_sentences(dataset_path):
    """Loads the dataset as a list of sentences.

    Params:
    - dataset_path (str): Path to dataset

    Returns:
    - sentences (pyspark.rdd.RDD): RDD containing sentences
    """
    data_frame = wordcount.sql_context.read.json(dataset_path)
    data_frame.show(n=5, truncate=100)
    records = data_frame.select(wordcount.TEXT_FIELD)
    # Filter out records shorter than 100 characters. Then,
    # add a unique ID to each record. Then,
    # make the ID the first element in the record, so it can be used as a key
    records = records.filter(F.length(F.col(wordcount.TEXT_FIELD)) > 99).rdd\
        .flatMap(lambda record: sent_tokenize(record[wordcount.TEXT_FIELD]))\
        .map(lambda record: Row(Sentences_t=record))\
        .zipWithUniqueId()\
        .map(lambda record: (record[1], record[0]))
    wordcount.rdd_show(records, "=====Loaded Sentences=====")
    return records
# End of load_sentences


def get_bag_of_words_labels(preprocessed_records, args):
    """Gets the labels for the bag of words. A label can be a a single important word, a collocation of two important
    words, or a set of synonyms of a word.

    Params:
    - preprocessed_records (pyspark.rdd.RDD): The tokenized, lemmatized, lowercase records
    - ars (argparse.Namespace): The command-line arguments passed to the program

    Returns:
    - bag_of_words_labels (list<str|tuple<str>>): The labels of the bag of words created
    """
    # frequent_words = wordcount.extract_frequent_words(preprocessed_records, args.num_words * 10, False)
    # frequent_words = dict(frequent_words)
    frequent_collocations = wordcount.extract_collocations(preprocessed_records, args.num_collocations,
                                                           args.collocation_window)
    tf_idf_scores = tfidf.tf_idf(preprocessed_records)
    # Pyspark technically ends here - the rest is processed on master node
    important_words = tfidf.extract_important_words(tf_idf_scores, args.num_words, False)
    # important_words_with_counts = synsets.add_word_counts(important_words, frequent_words)
    synset_dict = synsets.generate_syn_set(important_words)
    words_and_collocations = wordcount.merge_collocations_with_wordlist(frequent_collocations, important_words)
    # Merge words, collocations and synsets
    bag_of_words_labels = list()
    for item in words_and_collocations:
        if " " in item:  # item is a collocation
            bag_of_words_labels.append(item)
        elif item in synset_dict:  # item is an important word
            synset = synset_dict[item]
            if len(synset) == 1:  # synset only contains the word itself
                bag_of_words_labels.append(item)
            else:  # synset contains multiple words
                synset = [word.encode('utf-8') for word in synset[1:]]
                bag_of_words_labels.append(synset)
    # Save bag of words labels to single text file
    with open("bag_of_words_labels.json", "w") as bow_file:
        json.dump(bag_of_words_labels, bow_file)
    return bag_of_words_labels
# End of get_bag_of_words_labels()


def make_presence_feature_set(preprocessed_record, bag_of_words_labels):
    """Creates a feature set denoting the presence of features in the bag_of_words_labels. Results in an array (python
    list) that looks like this: [0.0, 1.0, 0.0, 0.0, ..., 0.0]. This is done by creating a feature set of feature counts
    using make_count_feature_set, and then clipping the counts at 1.0.

    Params:
    - preprocessed_record (list<str>): The tokenized, filtered and lemmatized record from which a feature set will be
                                       made
    - bag_of_words_labels (list<labels>): The list of bag of words labels, where a label can be a word, a bigram of
                                          two words separated by a space, or a synset

    Returns:
    - feature_set (list<float>): The feature set made using the preprocessed record and the bag of words labels
    """
    index, feature_counts = make_count_feature_set(preprocessed_record, bag_of_words_labels)
    feature_set = list()
    for count in feature_counts:
        if count > 0.0:
            feature_set.append(1.0)
        else:
            feature_set.append(0.0)
    return (index, feature_set)
# End of make_presence_feature_set()


def make_count_feature_set(preprocessed_record, bag_of_words_labels):
    """Creates a feature set denoting the count of features in the bag_of_words_labels. Results in an array (python
    list) that looks like this: [0.0, 2.0, 0.0, 4.0, ..., 1.0]

    Params:
    - preprocessed_record (list<str>): The tokenized, filtered and lemmatized record from which a feature set will be
                                       made
    - bag_of_words_labels (list<labels>): The list of bag of words labels, where a label can be a word, a bigram of
                                          two words separated by a space, or a synset

    Returns:
    - feature_set (list<float>): The feature set made using the preprocessed record and the bag of words labels
    """
    index, words_in_record = preprocessed_record
    bigrams_in_record = list()
    for word1, word2 in zip(words_in_record[:-1], words_in_record[1:]):
        bigrams_in_record.append(word1 + " " + word2)
    feature_set = list()
    for label in bag_of_words_labels:
        count = 0.0
        if isinstance(label, list) or isinstance(label, set) or isinstance(label, tuple):  # label represents a synset
            for word in label:
                if word in words_in_record:
                    count += 1.0
        elif " " in label:  # label represents a bigram
            for bigram in bigrams_in_record:
                if label == bigram:
                    count += 1.0
        else:  # label represents an important word
            for word in words_in_record:
                if word == label:
                    count += 1.0
        feature_set.append(count)
    return (index, feature_set)
# End of make_count_feature_set()


def create_dataset_from_feature_sets(records, preprocessed_records, presence_feature_set, count_feature_set):
    """Creates a dataset from the original records, preprocessed records, and the presence and count feature sets
    with a series of pyspark joins.

    The dataset will contain the following fields: 
    id, record, preprocessed_record, presence_feature_set, count_feature_set

    How it works:
    Multiple PySpark joins = a complicated mess
    dataset.join().map().join() overwrites the second element in each row
    Therefore, stuck with this nested tuple monstrosity
    row = (16, (((record, preprocessed_record), presence_feature_set), count_feature_set))
    row[1] = (((record, preprocessed_record), presence_feature_set), count_feature_set)
    row[1][0] = ((record, preprocessed_record), presence_feature_set)
    row[1][0][0] = (record, preprocessed_record)

    Params:
    - records (pyspark.rdd.RDD): The original records
    - preprocessed_records (pyspark.rdd.RDD): The preprocessed records
    - presence_feature_set (pyspark.rdd.RDD): The feature set representing the presence of each bag of words label
    - count_feature_set (pyspark.rdd.RDD): The feature set representing the count of each bag of words label

    Returns:
    - dataset (pyspark.rdd.RDD): The created dataset
    """
    dataset = records.map(lambda record: (record[constants.KEY], record[constants.VALUE][constants.TEXT]))\
        .join(preprocessed_records)\
        .join(presence_feature_set)\
        .join(count_feature_set)\
        .map(lambda row: (row[0], row[1][0][0][0], row[1][0][0][1], row[1][0][1], row[1][1]))
    return dataset
# End of create_dataset_from_feature_sets()


def save_dataset_as_dataframe(dataset, filename):
    """Converts the dataset RDD to a dataFrame and saves it as multiple JSON files.

    Params:
    - dataset (pyspark.rdd.RDD): The dataset containing the id, record, preprocessed record, and both feature sets for
                                 each record
    """
    data_frame = dataset.map(lambda record: Row(id=record[0], record=record[1], preprocessed_record=record[2],
                             presence_feature_set=record[3], count_feature_set=record[4]))
    data_frame = data_frame.toDF()
    data_frame.show()
    data_frame.write.json(filename, mode="overwrite")
# End of save_dataset_as_dataframe()


if __name__ == "__main__":
    args = parse_arguments()
    if args.sentences:
        sentences = load_sentences(args.file)
        with open("bag_of_words_labels.json", "r") as bow_file:
            bag_of_words_labels = json.load(bow_file)
        preprocessed_sentences = wordcount.preprocess_records(sentences)
        presence_feature_set = preprocessed_sentences.map(
            lambda record: make_presence_feature_set(record, bag_of_words_labels))
        count_feature_set = preprocessed_sentences.map(
            lambda record: make_count_feature_set(record, bag_of_words_labels))
        dataset_sentences = create_dataset_from_feature_sets(
            sentences, preprocessed_sentences, presence_feature_set, count_feature_set)
        wordcount.rdd_show(dataset_sentences, "=====Dataset=====")
        save_dataset_as_dataframe(dataset_sentences, args.output)
    else:
        records = wordcount.load_records(args.file, False)
        preprocessed_records = wordcount.preprocess_records(records)
        bag_of_words_labels = get_bag_of_words_labels(preprocessed_records, args)
        presence_feature_set = preprocessed_records.map(
            lambda record: make_presence_feature_set(record, bag_of_words_labels))
        count_feature_set = preprocessed_records.map(
            lambda record: make_count_feature_set(record, bag_of_words_labels))
        dataset = create_dataset_from_feature_sets(
            records, preprocessed_records, presence_feature_set, count_feature_set)
        wordcount.rdd_show(dataset, "=====Dataset=====")
        save_dataset_as_dataframe(dataset, args.output)
