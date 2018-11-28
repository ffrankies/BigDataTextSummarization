# coding: utf-8
"""Constructs bag-of-words-inspired feature sets out of the preprocessed records.
"""
import json
import argparse

import pyspark.sql.functions as F
from pyspark.sql.types import Row

import wordcount
import tfidf
import synsets
import constants
import os


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
    parser.add_argument('-s', '--sentences', action='store_true')
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
    records = wordcount.load_records(dataset_path)\
        .map(lambda record: Row(record_id=record[0], **record[1].asDict()))
    wordcount.rdd_show(records, "=====Records=====")
    
    # Filter out sentences shorter than 20 characters. Then,
    # add a unique ID to each record. Then,
    # make the ID the first element in the record, so it can be used as a key
    sentences = records.flatMap(record_to_sentences)\
        .filter(lambda sentence: len(sentence['Sentences_t']) > 19)\
        .zipWithUniqueId()\
        .map(lambda record: (record[1], record[0]))
    wordcount.rdd_show(sentences, "=====Loaded Sentences=====")
    return sentences
# End of load_sentences


def record_to_sentences(record):
    """Uses nltk to tokenize a record into sentences, keeping the original record's id.

    Params:
    - record (pyspark.sql.Row): The record to be tokenized

    Returns:
    - sentences (List[pyspark.sql.Row]): The list of tokenized sentences, with the record id
    """
    from nltk.tokenize import sent_tokenize

    record_id = record['record_id']
    sentences = sent_tokenize(record[wordcount.TEXT_FIELD])
    sentences = [Row(record_id=record_id, Sentences_t=sentence.encode('utf-8')) for sentence in sentences]
    return sentences
# End of record_to_sentences()


def preprocess_record(record, lemmatizer):
    """Preprocess a single record, keeping the record id.

    Params:
    - record (tuple[int, pyspark.sql.Row]): The record to preprocess
    - lemmatizer (wordnet.Lemmatizer): The lemmatizer model

    Returns:
    - record_id (int): The record id
    - contents (pyspark.sql.Row): The record contents, including the preprocessed record
    """
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    id, contents = record
    record_tokenized = word_tokenize(contents[wordcount.TEXT_FIELD])
    record_tagged = pos_tag(record_tokenized)
    record_filtered = wordcount.filter_stopwords((id, record_tagged))
    record_lemmatized = wordcount.lemmatize_record(record_filtered, lemmatizer)

    return Row(id=id, preprocessed_record=record_lemmatized[1], **contents.asDict())
# End of preprocess_record()


def preprocess_records_keep_fields(records):
    """Preprocess the given records, keeping any fields it has.

    Params:
    - records (pyspark.rdd.RDD): The RDD of records to preprocess

    Returns:
    - preprocessed_records (pyspark.rdd.RDD): The preprocessed records
    """
    lemmatizer = wordcount.WordNetLemmatizer()
    preprocessed_records = records.map(lambda record: preprocess_record(record, lemmatizer))
    wordcount.rdd_show(preprocessed_records, "=====Preprocessed Records=====")
    return preprocessed_records
# End of preprocess_records_keep_fields()


def get_bag_of_words_labels(preprocessed_records, args):
    """Gets the labels for the bag of words. A label can be a a single important word, a collocation of two important
    words, or a set of synonyms of a word.

    Params:
    - preprocessed_records (pyspark.rdd.RDD): The tokenized, lemmatized, lowercase records
    - ars (argparse.Namespace): The command-line arguments passed to the program

    Returns:
    - bag_of_words_labels (list<str|tuple<str>>): The labels of the bag of words created
    """
    reformatted_records = preprocessed_records.map(lambda record: (record['id'], record['preprocessed_record']))
    frequent_collocations = wordcount.extract_collocations(reformatted_records, args.num_collocations,
                                                           args.collocation_window)
    tf_idf_scores = tfidf.tf_idf(reformatted_records)
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


def make_feature_sets(preprocessed_record, bag_of_words_labels):
    """Creates the count and presence feature sets.

    Params:
    - preprocessed_record (pyspark.sql.Row): The tokenized, filtered and lemmatized record from which a feature set
                                             will be made
    - bag_of_words_labels (list<labels>): The list of bag of words labels, where a label can be a word, a bigram of
                                          two words separated by a space, or a synset

    Returns:
    - feature_set (pyspark.sql.Row): The record with both feature sets
    """
    feature_counts = make_count_feature_set(preprocessed_record, bag_of_words_labels)
    feature_set = make_presence_feature_set(feature_counts)
    return feature_set
# End of make_feature_sets()


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
    words_in_record = preprocessed_record['preprocessed_record']
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
    return Row(count_feature_set=feature_set, **preprocessed_record.asDict())
# End of make_count_feature_set()


def make_presence_feature_set(preprocessed_record):
    """Creates a feature set denoting the presence of features in the bag_of_words_labels. Results in an array (python
    list) that looks like this: [0.0, 1.0, 0.0, 0.0, ..., 0.0]. This is done by creating a feature set of feature counts
    using make_count_feature_set, and then clipping the counts at 1.0.

    Params:
    - preprocessed_record (pyspark.sql.RDD): The tokenized, filtered and lemmatized record from which a feature set
                                             will be made

    Returns:
    - feature_set (list<float>): The feature set made using the preprocessed record and the bag of words labels
    """
    feature_counts = preprocessed_record['count_feature_set']
    feature_set = list()
    for count in feature_counts:
        if count > 0.0:
            feature_set.append(1.0)
        else:
            feature_set.append(0.0)
    return Row(presence_feature_set=feature_set, **preprocessed_record.asDict())
# End of make_presence_feature_set()


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
    - dataset (pyspark.sql.DataFrame): The dataset containing the id, record, preprocessed record, and both feature
                                       sets for each record
    """
    dataset.show()
    dataset.write.json(filename, mode="overwrite")
# End of save_dataset_as_dataframe()


if __name__ == "__main__":
    args = parse_arguments()
    if args.sentences:
        sentences = load_sentences(args.file)
        with open("bag_of_words_labels.json", "r") as bow_file:
            bag_of_words_labels = json.load(bow_file)
        preprocessed_contents = preprocess_records_keep_fields(sentences)
    else:
        records = wordcount.load_records(args.file, False)
        preprocessed_contents = preprocess_records_keep_fields(records)
        if os.path.isfile("bag_of_words_labels.json"):
            print("Loading bag of words labels from file")
            with open("bag_of_words_labels.json", "r") as bow_file:
                bag_of_words_labels = json.load(bow_file)
        else:
            bag_of_words_labels = get_bag_of_words_labels(preprocessed_contents, args)
    feature_sets = preprocessed_contents.map(lambda contents: make_feature_sets(contents, bag_of_words_labels))
    dataset = feature_sets.toDF()
    save_dataset_as_dataframe(dataset, args.output)
