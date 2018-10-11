# coding: utf-8
"""Uses tf-idf to extract important words from a collection of documents.
"""

import math
from operator import add

import wordcount


def term_frequency(lemmatized_records):
    """Calculates the term frequency of words in the tokenized records. Term frequency is calculated as
    number of occurrences of a word / total number of words.

    NOTE: This gives really small term frequency values (~1.09 * 10^-5, in many cases), which could lead to loss of
    information.

    Params:
    - lemmatized_records (pyspark.rdd.RDD): The lemmatized and tokenized records

    Returns:
    - term_frequencies (dict): The term frequency of each word
    """
    lemma_counts = lemmatized_records.flatMap(lambda x: x)\
        .map(lambda x: (x, 1.0))\
        .reduceByKey(add)
    total_lemmas = lemma_counts.reduce(lambda x, y: ("", x[1] + y[1]))[1]
    term_frequencies = lemma_counts.map(lambda x: (x[0], x[1] / total_lemmas)).collect()
    term_frequencies = dict(term_frequencies)
    return term_frequencies
# End of term_frequency()


def inverse_document_frequency(lemmatized_records):
    """Calculates the inverse document frequency of the tokenized lemmas in the records. Inverse document frequency
    is calculated as log (number of records in which word appears / total number of records)

    Params:
    - lemmatized_records (list): The lemmatized versions of the records loaded from the JSON file
    - unique_words (list): The list of unique words in the records

    Returns:
    - inverse_document_frequencies (dict): The inverse document frequency of each word
    """
    num_records = float(lemmatized_records.count())
    inverse_document_frequencies = lemmatized_records.map(lambda record: set(record))\
        .flatMap(lambda record: record)\
        .map(lambda lemma: (lemma, 1.0))\
        .reduceByKey(add)\
        .filter(lambda df: df[1] > (0.1 * num_records))\
        .map(lambda df: (df[0], math.log(num_records / df[1], 10)))\
        .collect()
    inverse_document_frequencies = dict(inverse_document_frequencies)
    return inverse_document_frequencies
# End of inverse_document_frequency()


def tf_idf(records):
    """Calculates the term frequency inverse document frequency for each lemmatized non-stopword in the dataset of
    records.

    Params:
    - records (list): The records loaded from the dataset

    Returns:
    - tf_idf_scores (list): The tf-idf score for each lemmatized non-stopword in the dataset
    """
    term_frequencies = term_frequency(records)
    inverse_document_frequencies = inverse_document_frequency(records)
    tf_idf_scores = list()
    for word in inverse_document_frequencies.keys():
        tf_idf_scores.append((word, term_frequencies[word] * inverse_document_frequencies[word]))
    return tf_idf_scores
# End of tf_idf()


def extract_important_words(tf_idf_scores, num_words, strip_scores=True):
    """Extracts important words based on their tf_idf_scores.

    Params:
    - tf_idf_scores (list): The tf-idf score of each word in the document
    - num_words (int): The number of words to extract
    - strip_scores (bool): If set to True, will split TF-IDF scores from the result

    Returns:
    - important_words (list): The list of words with the highest tf-idf score
    """
    important_words = sorted(tf_idf_scores, key=lambda score: score[1], reverse=True)
    if strip_scores:
        important_words = [score[0] for score in important_words[:num_words]]
    return important_words
# End of extract_important_words()


if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file, False)
    records = wordcount.preprocess_records(records)
    tf_idf_scores = tf_idf(records)
    # Pyspark technically ends here - the rest is processed on master node
    important_words = extract_important_words(tf_idf_scores, args.num_words, False)
    print("=====Important Words Identified by TF-IDF=====")
    print(important_words)
    collocations = wordcount.extract_collocations(records, args.num_collocations, args.collocation_window)
    collocations = [collocation[0] for collocation in collocations]
    words_and_collocations = wordcount.merge_collocations_with_wordlist(collocations, important_words)
    print("=====Important Words and Collocations=====")
    print(words_and_collocations)
