"""Automatic labeling of records based off of basic statistics about the feature set.
"""
import argparse

import numpy as np

import pyspark
from pyspark.sql import Row

spark_context = pyspark.SparkContext.getOrCreate()
sql_context = pyspark.SQLContext(spark_context)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='The dataset to classify')
    parser.add_argument('-o', '--output', type=str, help='The output file containing relevant records')
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


def automatically_label_dataset(feature_count_dataset_rdd, feature_count_mean, feature_count_sd):
    """Automatically labels the records in the dataset using the following rules:

    - Relevent records are labeled with 1.0, irrelevant records are labeled with 0.0, maybes are labeled with -1.0
    - A record is relevant if feature count > feature_count_mean + 2 * feature_count_sd
    - A record is irrelevant if feature count <= 1

    Params:
    - feature_count_dataset_rdd (pyspark.rdd.RDD): The dataset, in RDD format, with a 'feature_count' column
    - feature_count_mean (float): The mean of the feature counts
    - feature_count_sd (float): The standard deviation of the feature counts

    Returns:
    - autolabeled_dataset (pyspark.rdd.RDD): The dataset, with automatic record labels
    """
    irrelevant_threshold = 2.0
    relevant_threshold = feature_count_mean + (2 * feature_count_sd)
    autolabeled_dataset = feature_count_dataset_rdd\
        .map(lambda row: automatically_label_record(row, irrelevant_threshold, relevant_threshold))
    return autolabeled_dataset
# End of automatically_label_dataset()


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
    dataset = automatically_label_dataset(dataset, feature_count_mean, feature_count_sd)
    save_autolabeled_dataset(dataset, args.output)
