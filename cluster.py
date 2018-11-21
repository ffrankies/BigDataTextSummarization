# -*- coding: utf-8 -*-
"""Clusters the items in the provided RDD using various methods and Bag-of-Words.
"""

import argparse

import pyspark
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Tokenizer
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml.classification import MultilayerPerceptronClassifier

from numpy import argmax

import wordcount

spark_context = pyspark.SparkContext.getOrCreate()
spark_context.setLogLevel("OFF")
sql_context = pyspark.SQLContext(spark_context)


#####################
# FEATURE TYPES
#####################
FEATURE_COUNT = 'count'
FEATURE_TFIDF = 'tfidf'

#####################
# CLUSTERING METHOD
#####################
METHOD_KMEANS = 'kmeans'
METHOD_BISECTING_KMEANS = 'bisecting'
METHOD_LDA = 'lda'


def parse_arguments():
    """Parses command-line arguments.

    Returns:
    - args (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='The dataset to classify')
    parser.add_argument('-o', '--output', type=str, help='The output file containing relevant records')
    parser.add_argument('-n', '--num_clusters', type=int, default=5, help='The number of clusters to use')
    parser.add_argument('-m', '--method', type=str, default='kmeans', help='The clustering method to use')
    parser.add_argument('-f', '--feature_type', type=str, default='count', help='The features to use for clustering')
    parser.add_argument('-v', '--vocab_size', type=int, default=10000, help='The vocabulary size to be considered')
    return parser.parse_args()
# End of parse_arguments()


def load_dataset(dataset_dir):
    """Loads the dataset from the dataset directory. Also converts the features into Vectors.

    Params:
    - dataset_dir (str): Directory where spark stored dataset

    Returns:
    - rdd (pyspark.rdd.RDD): The dataset, converted to an RDD
    """
    data_frame = sql_context.read.json(dataset_dir)
    data_frame.show()
    return data_frame.rdd
# End of load_dataset()


def compute_features(dataset, feature_type, feature_size):
    """Computes the clustering features from the dataset based on the given feature type.

    Params:
    - dataset (pyspark.rdd.RDD): The loaded dataset
    - feature_type (str): The type of features to generate
    - feature_size (int): The number of features to consider

    Returns:
    - features (pyspark.sql.DataFrame): The data frame containing computed features
    """
    if feature_type == FEATURE_TFIDF:
        features = tfidf(dataset, feature_size)
    elif feature_type == FEATURE_COUNT:
        features = count_vectorize(dataset, feature_size)
    else:
        raise NotImplementedError("{} is not a valid feature type".format(feature_type))
    return features
# End of compute_features()


def count_vectorize(dataset, vocab_size):
    """Uses the CountVectorizer to create a "bag of words" for the dataset items.

    Params:
    - dataset (pyspark.rdd.RDD): The dataset to be vectorized
    - vocab_size (int): The size of the vocabulary to be considered

    Returns:
    - vectorized (pyspark.sql.DataFrame): The vectorized dataset
    """
    from nltk.tokenize import word_tokenize
    dataset = dataset.map(lambda row: Row(words=word_tokenize(row['Sentences_t']), **row.asDict()))
    data_frame = dataset.toDF()
    min_df = int(dataset.count() * 0.1)
    count_vectorizer = CountVectorizer(inputCol='words', outputCol='features', vocabSize=vocab_size, minDF=min_df)
    count_vectorizer_model = count_vectorizer.fit(data_frame)
    vectorized = count_vectorizer_model.transform(data_frame)
    vectorized.show()
    return vectorized
# End of count_vectorize()


def tfidf(dataset, vocab_size):
    """Computes the tf-idf scores for the given dataset.

    Params:
    - dataset (pyspark.rdd.RDD): The dataset to be vectorized
    - vocab_size (int): The size of the vocabulary to be considered

    Returns:
    - tfidf_scores (pyspark.sql.DataFrame): The tfidf scores for the dataset
    """
    tokenizer = Tokenizer(inputCol="Sentences_t", outputCol="words")
    words = tokenizer.transform(dataset.toDF())

    hashingTF = HashingTF(numFeatures=vocab_size, inputCol="words", outputCol="tf")
    tf = hashingTF.transform(words)

    min_df = int(dataset.count() * 0.1)
    idf = IDF(inputCol="tf", outputCol="features", minDocFreq=min_df)
    idfModel = idf.fit(tf)
    tfidf_scores = idfModel.transform(tf).drop('words').drop('tf')
    tfidf_scores.show()
    return tfidf_scores
# End of tfidf()


def do_cluster(features, cluster_method, num_clusters):
    """Does the clustering on the features dataset. Assumes a column named 'features'

    Params:
    - features (pyspark.sql.DataFrame): The data frame containing the features to be used for clustering
    - cluster_method (str): The method to be used for clustering
    - num_clusters (int): The number of clusters to be used

    Returns:
    - clustered (pyspark.sql.DataFrame): The data frame, with the predicted clusters in a 'cluster' column
    """
    if cluster_method == METHOD_BISECTING_KMEANS:
        raise NotImplementedError('Bisecting Kmeans clustering not implemented yet')
    elif cluster_method == METHOD_LDA:
        clustered = lda(features, num_clusters)
    elif cluster_method == METHOD_KMEANS:
        clustered = kmeans(features, num_clusters)
    else:
        raise NotImplementedError("{} is not a valid clustering method".format(cluster_method))
    return clustered
# End of do_cluster()


def kmeans(features, num_clusters):
    """Does clustering on the features dataset using KMeans clustering.

    Params:
    - features (pyspark.sql.DataFrame): The data frame containing the features to be used for clustering
    - num_clusters (int): The number of clusters to be used

    Returns:
    - clustered (pyspark.sql.DataFrame): The data frame, with the predicted clusters in a 'cluster' column
    """
    kmeans = KMeans(k=num_clusters, featuresCol='features', predictionCol='cluster')
    kmeans_model = kmeans.fit(features)
    clustered = kmeans_model.transform(features)
    clustered.show()
    print("=====Clustering Results=====")
    print("Clustering cost = ", kmeans_model.computeCost(features))
    print("Cluster sizes = ", kmeans_model.summary.clusterSizes)
    return clustered
# End of kmeans()


def lda(features, num_clusters):
    """Does clustering on the features dataset using LDA topic clustering.

    Params:
    - features (pyspark.sql.DataFrame): The data frame containing the features to be used for clustering
    - num_clusters (int): The number of clusters to be used

    Returns:
    - clustered (pyspark.sql.DataFrame): The data frame, with the predicted clusters in a 'cluster' column
    """
    lda = LDA(k=num_clusters, featuresCol='features', topicDistributionCol='topics')
    lda_model = lda.fit(features)
    clustered = lda_model.transform(features)
    clustered = clustered.rdd.map(lambda row: Row(cluster=int(argmax(row['topics'])), **row.asDict())).toDF()
    clustered = clustered.drop('topics')
    clustered.show()
    print("=====Clustering Results=====")
    print("LDA log perplexity = ", lda_model.logPerplexity(features))
    cluster_sizes = list()
    for i in range(num_clusters):
        cluster_size = clustered.rdd.filter(lambda row: row['cluster'] == i).count()
        cluster_sizes.append(cluster_size)
    print("Cluster sizes = ", cluster_sizes)
    # Do an argmax over the clusters to get the actual topic, I guess
    return clustered
# End of lda()


if __name__ == "__main__":
    args = parse_arguments()
    print("Arguments: ", args)
    dataset = load_dataset(args.dataset)
    dataset = compute_features(dataset, args.feature_type, args.vocab_size)
    dataset = do_cluster(dataset, args.method,  args.num_clusters)
