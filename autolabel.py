"""Automatic labeling of records based off of basic statistics about the feature set.
"""
import argparse

import numpy as np

import pyspark
from pyspark.sql import Row

spark_context = pyspark.SparkContext.getOrCreate()
spark_context.setLogLevel("OFF")
sql_context = pyspark.SQLContext(spark_context)


def parse_arguments():
    """Parses the command line arguments provided by the user.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='The dataset to classify')
    parser.add_argument('-o', '--output', type=str, help='The output file containing relevant records')
    parser.add_argument('-i', '--irrelevant_t', type=float, default=2.0,
                        help='The threshold below which record is irrelevant')
    parser.add_argument('-r', '--relevant_t', type=float, default=2.0,
                        help='Record is relevant if feature count > mean + relevant_t * s.d.')
    parser.add_argument('-s', '--contains_sentences', action='store_true',
                        help='Add this argument if the dataset consists of sentences and not records.')
    return parser.parse_args()
# End of parse_arguments()


def read_dataset_as_rdd(dataset_path):
    """Reads the dataset file as an RDD.

    Params:
    - dataset_path (str): The path to the dataset. If running on Hadoop, make this a Hadoop path.

    Returns:
    - dataset_rdd (pyspark.rdd.RDD): The loaded dataset, in RDD format for easy manipulation
    """
    dataset_data_frame = sql_context.read.json(dataset_path)
    dataset_rdd = dataset_data_frame.rdd
    return dataset_rdd
# End of read_dataset_as_rdd()


def add_feature_counts_column(dataset_rdd):
    """Add a 'feature_count' column to the dataset RDD. This is calculated as the sum of the values in the
    'presence_feature_set' column.

    Params:
    - dataset_rdd (pyspark.rdd.RDD): The dataset, in RDD format

    Returns:
    - feature_count_dataset_rdd (pyspark.rdd.RDD): The dataset, in RDD format, with a 'feature_count' column
    """
    feature_count_dataset_rdd = dataset_rdd\
        .map(lambda row: Row(feature_count=sum(row['presence_feature_set']), **row.asDict()))
    return feature_count_dataset_rdd
# End of add_feature_counts_column()


def get_feature_counts_statistics(feature_count_dataset_rdd):
    """Calculates the mean and standard deviation of the feature counts in the 'feature_count' column.

    Params:
    - feature_count_dataset_rdd (pyspark.rdd.RDD): The dataset, in RDD format, with a 'feature_count' column

    Returns:
    - feature_count_mean (float): The mean of the feature counts
    - feature_count_sd (float): The standard deviation of the feature counts
    """
    feature_counts = feature_count_dataset_rdd.map(lambda row: row['feature_count']).collect()
    feature_count_mean = np.mean(feature_counts)
    feature_count_sd = np.std(feature_counts)
    return feature_count_mean, feature_count_sd
# End of get_feature_counts_statistics()


def automatically_label_dataset(feature_count_dataset_rdd, feature_count_mean, feature_count_sd, irrelevant_t, 
    relevant_t):
    """Automatically labels the records in the dataset using the following rules:

    - Relevent records are labeled with 1.0, irrelevant records are labeled with 0.0, maybes are labeled with -1.0
    - A record is relevant if feature count > feature_count_mean + 2 * feature_count_sd
    - A record is irrelevant if feature count <= 1

    Params:
    - feature_count_dataset_rdd (pyspark.rdd.RDD): The dataset, in RDD format, with a 'feature_count' column
    - feature_count_mean (float): The mean of the feature counts
    - feature_count_sd (float): The standard deviation of the feature counts
    - irrelevant_t (float): If feature count < irrelevant_t, then the record is irrelevant
    - relevant_t (float): If feature count > feature_count_mean + relevant_t * feature_count_sd, record is relevant

    Returns:
    - autolabeled_dataset (pyspark.rdd.RDD): The dataset, with automatic record labels
    """
    irrelevant_threshold = irrelevant_t
    relevant_threshold = feature_count_mean + (relevant_t * feature_count_sd)
    print("Relevant threshold = %.2f | Relevant threshold = %.2f" % (relevant_threshold, irrelevant_threshold))
    autolabeled_dataset = feature_count_dataset_rdd\
        .map(lambda row: automatically_label_record(row, irrelevant_threshold, relevant_threshold))
    return autolabeled_dataset
# End of automatically_label_dataset()


def automatically_label_record(record_row, irrelevant_threshold, relevant_threshold):
    """Automatically label the record based on the following rules:

    - Relevent records are labeled with 1.0, irrelevant records are labeled with 0.0, maybes are labeled with -1.0
    - A record is relevant if feature count > relevant_threshold
    - A record is irrelevant if feature count < irrelevant_threshold

    Labeling is done by adding a 'label' column to the row, with the auto-generated label.

    Params:
    - record_row (pyspark.sql.Row): The record to be labeled
    - irrelevant_threshold (float): The threshold beneath which records are considered irrelevant
    - relevant_threshold (float): The threshold above which records are considered relevant

    Returns:
    - labeled_row (pyspark.sql.Row): The record, with the automatically generated label in the 'label' column
    """
    feature_count = record_row['feature_count']
    generated_label = -1.0
    if feature_count < irrelevant_threshold:
        generated_label = 0.0
    if feature_count > relevant_threshold:
        generated_label = 1.0
    labeled_row = Row(label=generated_label, **record_row.asDict())
    return labeled_row
# End of automatically_label_record()


def augment_automatic_labels(dataset, contains_sentences):
    """Augment automatic labeling with some rules.

    For labeling records, any record that does not contain the bigram "Hurricane Florence" is labeled as irrelevant.

    For labeling sentences, any sentence containing any of the ff words/bigrams was labeled as irrelevant: 
    "picture", "all rights reserved", "http", "tap here", "pictured".

    Params:
    - dataset (pyspark.rdd.RDD): The automatically labeled dataset
    - contains_sentences (bool): If True, dataset contains sentences. Otherwise, it contains records

    Returns:
    - augmented_autolabeled_dataset (pyspark.rdd.RDD): The automatically labeled dataset, with augmented labels
    """
    if contains_sentences:
        augmented_autolabeled_dataset = dataset.map(mark_irrelevant_sentence)
    else:  # contains records
        augmented_autolabeled_dataset = dataset.map(mark_irrelevant_record)
    return augmented_autolabeled_dataset
# End of augment_automatic_labels()


def mark_irrelevant_record(record):
    """Marks a record as irrelevant if it does not contain the bigram "Hurricane Florence"
    
    Params:
    - record (pyspark.sql.Row): The record to mark as irrelevant

    Returns:
    - marked_record (pyspark.sql.Row): The marked record
    """
    record_dictionary = record.asDict()
    if 'hurricane florence' not in record_dictionary['Sentences_t'].lower():
        record_dictionary['label'] = 0.0
        print("Augmented a record label")
    return Row(**record_dictionary)
# End of mark_irrelevant_record()


def mark_irrelevant_sentence(sentence):
    """Marks a sentence as irrelevant if it contains any of the following words or bigrams: 
    "picture", "all rights reserved", "http", "tap here", "pictured".
    
    Params:
    - sentence (pyspark.sql.Row): The sentence to mark as irrelevant

    Returns:
    - marked_sentence (pyspark.sql.Row): The marked sentence
    """
    print("Marking irrelevant sentence")
    FILTER_FROM = ['picture', 'all rights reserved', 'http', 'tap here', 'pictured', 'photo', 'gallery', 'galleries',
                   'share', 'facebook', '.com']
    sentence_dictionary = sentence.asDict()
    for filter_token in FILTER_FROM:
        sentences = sentence_dictionary['Sentences_t'].lower()
        if filter_token in sentences:
            print("Sentence is irrelevant")
            sentence_dictionary['label'] = 0.0
            break
        if len(sentences) < 20 or len(sentences) > 400:
            print("Sentence is irrelevant")
            sentence_dictionary['label'] = 0.0
            break
    return Row(filtered=True, **sentence_dictionary)
# End of mark_irrelevant_sentence()


def autolabeled_statistics(autolabeled_dataset):
    """Prints the autolabeling statistics.

    Params:
    - autolabeled_dataset (pyspark.rdd.RDD): The automatically labeled dataset
    """
    num_positive = autolabeled_dataset.filter(lambda record: record['label'] == 1.0).count()
    num_negative = autolabeled_dataset.filter(lambda record: record['label'] == 0.0).count()
    print("====Autolabeled Statistics====")
    print("Number of positive records | number of negative records | total records = ", num_positive, " | ", 
          num_negative, " | ", autolabeled_dataset.count())
# End of autolabeled_statistics()


def save_autolabeled_dataset(autolabeled_dataset, output_filename):
    """Save the autolabeled dataset as a series of JSON files. First converts the RDD to a dataframe, because most
    pyspark Machine Learning approaches seem to use data frames (and also data frames are easier to save).

    Dataset JSON files will be stored in the "autolabeled_dataset" directory on Hadoop if run on cluster, or in the
    current directory if run locally.

    Params:
    - autolabeled_dataset (pyspark.rdd.RDD): The automatically labeled dataset.
    """
    dataset_data_frame = autolabeled_dataset.toDF()
    dataset_data_frame.show()  # Show what we're saving
    dataset_data_frame.write.json(output_filename, mode="overwrite")
# End of save_autolabeled_dataset()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = read_dataset_as_rdd(args.dataset)
    dataset = add_feature_counts_column(dataset)
    feature_count_mean, feature_count_sd = get_feature_counts_statistics(dataset)
    dataset = automatically_label_dataset(dataset, feature_count_mean, feature_count_sd,
                                          args.irrelevant_t, args.relevant_t)
    autolabeled_statistics(dataset)
    dataset = augment_automatic_labels(dataset, args.contains_sentences)
    autolabeled_statistics(dataset)
    save_autolabeled_dataset(dataset, args.output)
