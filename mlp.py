# coding: utf-8
"""Uses a multi-layer perceptron to classify records in a data frame as either relevant or irrelevant.
"""

import argparse
import random
import json
import math

import pyspark
from pyspark.sql import Row
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassifier

import wordcount

spark_context = pyspark.SparkContext.getOrCreate()
spark_context.setLogLevel("OFF")
sql_context = pyspark.SQLContext(spark_context)

def parse_arguments():
    """Parses command-line arguments.

    Returns:
    - args (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='The dataset to classify')
    parser.add_argument('-o', '--output', type=str, help='The output file containing relevant records')
    parser.add_argument('-s', '--combined_sentence_output', type=str, 
                        help='If provided, sentences will be combined into records consisting of relevant sentences.')
    return parser.parse_args()
# End of parse_arguments()


def load_dataset(dataset_dir):
    """Loads the dataset from the dataset directory. Also converts the features into Vectors.

    Params:
    - dataset_dir (str): Directory where spark stored dataset

    Returns:
    - rdd_vector_features (pyspark.rdd.RDD): The loaded dataset, with features as Vectors
    """
    data_frame = sql_context.read.json(dataset_dir)
    data_frame.show()
    if 'URL_s' in data_frame.columns:
        print("URLs are present in data frame")
        if 'record_id' in data_frame.columns:
            rdd_vector_features = data_frame.rdd.map(lambda row: Row(
                count_feature_set=Vectors.dense(row['count_feature_set']), 
                presence_feature_set=Vectors.dense(row['presence_feature_set']),
                feature_count=row['feature_count'],
                id=row['id'],
                label=row['label'],
                preprocessed_record=row['preprocessed_record'],
                record=row['Sentences_t'].encode('utf-8'),
                record_id=row['record_id'],
                URL_s=row['URL_s']
            ))
        else:
            rdd_vector_features = data_frame.rdd.map(lambda row: Row(
                count_feature_set=Vectors.dense(row['count_feature_set']), 
                presence_feature_set=Vectors.dense(row['presence_feature_set']),
                feature_count=row['feature_count'],
                id=row['id'],
                label=row['label'],
                preprocessed_record=row['preprocessed_record'],
                record=row['Sentences_t'].encode('utf-8'),
                URL_s=row['URL_s']
            ))
    else:
        if 'record_id' in data_frame.columns:
            rdd_vector_features = data_frame.rdd.map(lambda row: Row(
                count_feature_set=Vectors.dense(row['count_feature_set']), 
                presence_feature_set=Vectors.dense(row['presence_feature_set']),
                feature_count=row['feature_count'],
                id=row['id'],
                label=row['label'],
                preprocessed_record=row['preprocessed_record'],
                record=row['Sentences_t'].encode('utf-8'),
                record_id=row['record_id']
            ))
        else:
            rdd_vector_features = data_frame.rdd.map(lambda row: Row(
                count_feature_set=Vectors.dense(row['count_feature_set']), 
                presence_feature_set=Vectors.dense(row['presence_feature_set']),
                feature_count=row['feature_count'],
                id=row['id'],
                label=row['label'],
                preprocessed_record=row['preprocessed_record'],
                record=row['Sentences_t'].encode('utf-8')
            ))
    return rdd_vector_features
# End of load_dataset()


def extract_dataset_partitions(dataset):
    """Extracts the training and testing partitions from the dataset.

    Params:
    - dataset (pyspark.rdd.RDD): The dataset whose partitions are to be extracted

    Returns:
    - train (pyspark.rdd.RDD): The training partition
    - test (pyspark.rdd.RDD): The testing partition
    - num_features (int): The number of features
    """
    labeled_data = dataset.filter(lambda row: row['label'] == 0.0 or row['label'] == 1.0)
    train_data, test_data = labeled_data.randomSplit([0.7, 0.3])
    train_data, test_data = undersample(train_data, test_data)
    num_features = len(train_data.take(1)[0]['presence_feature_set'])
    train_data = train_data.toDF()
    test_data = test_data.toDF()
    return train_data, test_data, num_features
# End of extract_dataset_partitions()


def undersample(train_data, test_data):
    """Performs undersampling to account for the discrepancy b/n numbers of positive and negative training samples.

    Params:
    - train_data (pyspark.rdd.RDD): The training data
    - test_data (pyspark.rdd.RDD): The testing data

    Returns:
    - train_data (pyspark.rdd.RDD): The training data
    - test_data (pyspark.rdd.RDD): The testing data
    """
    print("Before undersampling, num train samples = %d, num test samples = %d" % 
          (train_data.count(), test_data.count()))
    positive_training_samples = train_data.filter(lambda row: row['label'] == 1.0)
    negative_training_samples = train_data.filter(lambda row: row['label'] == 0.0)
    num_positive_training_samples = positive_training_samples.count()
    num_negative_training_samples = negative_training_samples.count()
    if num_negative_training_samples == num_positive_training_samples:
        return train_data, test_data
    minimum = min([num_negative_training_samples, num_positive_training_samples])
    maximum = max([num_negative_training_samples, num_positive_training_samples])
    ratio_one = float(minimum) / float(maximum)
    ratio_two = 1.0 - ratio_one
    if positive_training_samples.count() > negative_training_samples.count():
        positive_training_samples, test_additions = positive_training_samples.randomSplit([ratio_one, ratio_two])
    else:
        negative_training_samples, test_additions = negative_training_samples.randomSplit([ratio_one, ratio_two])
    test_data = test_data.union(test_additions)
    train_data = positive_training_samples.union(negative_training_samples)
    print("After undersampling, num train samples = %d, num test samples = %d" % 
          (train_data.count(), test_data.count()))
    return train_data, test_data
# End of undersample()


def train_model(train_data, num_features):
    """Train the multilayer perceptron model.

    Params:
    - train_data (pyspark.rdd.RDD): The training dataset partition
    - num_features (int): The number of features

    Returns:
    - model (pyspark.ml.MultilayerPerceptronModel): The trained MLP model
    """
    multilayer_perceptron = MultilayerPerceptronClassifier(
        blockSize=1,
        featuresCol="presence_feature_set",
        labelCol="label",
        predictionCol="prediction",
        layers=[num_features, 100, 50, 10, 2]
    )
    model = multilayer_perceptron.fit(train_data)
    return model
# End of train_model()


def model_accuracy(model, test_data):
    """Finds and prints the accuracy of the trained model on the test partition.

    Params:
    - model (pyspark.ml.MultilayerPerceptronModel): The trained MLP model
    - test_data (pyspark.rdd.RDD): The test partition of the dataset
    """
    result = model.transform(test_data)
    predictionAndLabels = result.select("prediction", "label")

    def check_correct(row):
        if row['prediction'] == row['label']:
            return 1
        else:
            return 0

    checker = predictionAndLabels.rdd.map(lambda row: check_correct(row))
    num_correct = float(checker.reduce(lambda a,b: a + b))
    num_total = float(test_data.count())
    print(str(num_correct) + "/" + str(num_total))
# End of model_accuracy()


def predict_and_separate(dataset, model):
    """Uses the trained model to make predictions on the unlabeled portions of the dataset, as well as separates
    the relevant and irrelevant records.

    Params:
    - dataset (pyspark.rdd.RDD): The loaded dataset
    - model (pyspark.ml.MultilayerPerceptronModel): The trained MLP model

    Returns:
    - relevant (pyspark.rdd.RDD): The predicted relevant records
    - irrelevant (pyspark.rdd.RDD): The predicted irrelevant records
    """
    unlabeled_data = dataset.filter(lambda row: row['label'] == -1).toDF()
    predicted = model.transform(unlabeled_data)
    important_feature_selector(predicted)
    result = predicted.select("record", "prediction").rdd
    positive = result.filter(lambda row: row['prediction'] == 1.0).take(10)
    print("=====Positive=====")
    for pos in positive:
        print(pos)
    negative = result.filter(lambda row: row['prediction'] == 0.0).take(10)
    print("=====Negative=====")
    for neg in negative:
        print(neg)

    # Get relevant records
    labeled_pos = dataset.filter(lambda row: row['label'] == 1.0).collect()  # list of Rows
    unlabeled_pos = predicted.drop('label').withColumnRenamed('prediction', 'label')\
        .rdd.filter(lambda row: row['label'] == 1.0).collect()  # list of Rows
    labeled_pos.extend(unlabeled_pos)
    print("Number of relevant records = ", len(labeled_pos), " out of: ", dataset.count())
    relevant = spark_context.parallelize(labeled_pos)

    # Get irrelevant records
    labeled_neg = dataset.filter(lambda row: row['label'] == 0.0).collect()  # list of Rows
    unlabeled_neg = predicted.drop('label').withColumnRenamed('prediction', 'label')\
        .rdd.filter(lambda row: row['label'] == 0.0).collect()  # list of Rows
    labeled_neg.extend(unlabeled_neg)
    print("Number of irrelevant records = ", len(labeled_neg), " out of: ", dataset.count())
    irrelevant = spark_context.parallelize(labeled_neg)

    return relevant, irrelevant
# End of predict_and_separate()


def important_feature_selector(predicted):
    """Uses the Chi-Squared Test to select important features for classification, and prints them out.
    
    Params:
    - predicted (pyspark.sql.DataFrame): The dataset, with predictions
    """
    selector = ChiSqSelector(
        numTopFeatures=50, 
        featuresCol='presence_feature_set', 
        labelCol='label', 
        outputCol='selected_features', 
        selectorType='numTopFeatures')
    model = selector.fit(predicted)
    important_features = model.selectedFeatures
    with open('bag_of_words_labels.json', 'r') as bow_file:
        bow_labels = json.loads(bow_file.readlines()[0])  # There is only one line
    important_feature_labels = [bow_labels[index] for index in important_features]
    print("=====Important Feature Labels=====")
    print(important_feature_labels)
# End of important_feature_selector()


def examine_predictions(relevant, irrelevant):
    """Examines the predictions made.

    Params:
    - relevant (pyspark.rdd.RDD): The predicted relevant records
    - irrelevant (pyspark.rdd.RDD): The predicted irrelevant records
    """
    import numpy as np
    pos_counts = relevant.filter(lambda row: row['label'] == 1.0)\
        .map(lambda row: sum(row['presence_feature_set'])).collect()
    neg_counts = irrelevant.filter(lambda row: row['label'] == 0.0)\
        .map(lambda row: sum(row['presence_feature_set'])).collect()
    avg_count_pos = np.mean(pos_counts)
    avg_count_neg = np.mean(neg_counts)
    print(avg_count_pos, avg_count_neg)
    min_count_pos = np.min(pos_counts)
    max_count_neg = np.max(neg_counts)
    print(min_count_pos, max_count_neg)
# End of examine_predictions()


def save_relevant_records(relevant, output_filename, combined_sentence_output):
    """Save relevant records to file.

    Params:
    - relevant (pyspark.rdd.RDD): The predicted relevant records
    - output_filename (str): The name of the output file
    - combined_sentence_output (str): The name of the output file for the combined relevant sentences
    """
    def utf_encode_row(row):
        """Forces UTF-8 encoding on the Sentences_t field of the row.
        """
        return Row(Sentences_t=row['record'].encode('utf-8'), **row.asDict())
    
    relevant = relevant.map(utf_encode_row)
    data_frame = relevant.toDF()\
        .drop(*['count_feature_set', 'feature_count', 'label', 'presence_feature_set', 'record'])
    data_frame.show()
    data_frame.write.json(output_filename, mode="overwrite")
    combine_sentence_output(combined_sentence_output, data_frame)
# End of save_relevant_records


def combine_sentence_output(combined_sentence_output, sentences):
    """Combines relevant sentences into records containing only relevant sentences and saves the result. If 
    combined_sentence_output is empty, does nothing. 

    Params:
    - combined_sentence_output (str): The name of the output file for the combined relevant sentences
    - sentences (pyspark.sql.DataFrame): The data frame to combine
    """
    if not combined_sentence_output:
        return
    if 'URL_s' in sentences.columns:
        sentences_ids = sentences.rdd.map(
            lambda sentence: (sentence['record_id'], [[sentence['Sentences_t']], sentence['URL_s']])
        ).collect()
    else:
        sentences_ids = sentences.rdd.map(
            lambda sentence: (sentence['record_id'], [[sentence['Sentences_t']]])
        ).collect()
    sentences_aggregated = dict()
    for key, contents in sentences_ids:  # Contents is a single sentence, with its article's URL
        if key in sentences_aggregated:
            if contents[0][0] not in sentences_aggregated[key]:  # Prevent duplicates
                # print(contents[0], " not in ", sentences_aggregated[key])
                sentences_aggregated[key][0].append(contents[0][0])
        else:
            sentences_aggregated[key] = contents
    print("=====Sentences Aggregated=====")
    print(sentences_aggregated.items()[:5])
    reconstructed_records = list()
    for key, relevant_sentences in sentences_aggregated.items():
        if len(relevant_sentences[0]) < 5:  # Skip over reconstructed record with less than 5 sentences
            continue
        # reconstructed_record = list(map(lambda sentence: sentence['Sentences_t'], relevant_sentences))
        reconstructed_record = " ".join(relevant_sentences[0])
        if 'URL_s' in sentences.columns:
            reconstructed_record_row = Row(Sentences_t=reconstructed_record, URL_s=relevant_sentences[1])
        else:
            reconstructed_record_row = Row(Sentences_t=reconstructed_record)
        reconstructed_records.append(reconstructed_record_row)
    reconstructed_records = spark_context.parallelize(reconstructed_records)
    wordcount.rdd_show(reconstructed_records, "====Reconstructed Records====")
    print("Number of reconstructed records = %d" % reconstructed_records.count())
    reconstructed_records.toDF().write.json(combined_sentence_output, "overwrite")
# End of combine_sentence_output()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load_dataset(args.dataset)
    train, test, num_features = extract_dataset_partitions(dataset)
    model = train_model(train, num_features)
    model_accuracy(model, test)
    relevant, irrelevant = predict_and_separate(dataset, model)
    examine_predictions(relevant, irrelevant)
    save_relevant_records(relevant, args.output, args.combined_sentence_output)
