# coding: utf-8
"""Uses tf-idf to extract important words from a collection of documents.
"""

import math

from nltk.probability import FreqDist

import wordcount
import constants


def term_frequency(tokenized_records):
    """Calculates the term frequency of words in the tokenized records. Term frequency is calculated as
    number of occurrences of a word / total number of words.

    NOTE: This gives really small term frequency values (~1.09 * 10^-5, in many cases), which could lead to loss of
    information.

    Params:
    - tokenized_records (list): The lemmatized and tokenized words from the records

    Returns:
    - term_frequencies (dict): The term frequency of each word
    """
    word_counts = FreqDist(tokenized_records)
    term_frequencies = dict(word_counts)
    unique_words = term_frequencies.keys()
    num_words = float(len(tokenized_records))
    for word in unique_words:
        term_frequencies[word] = float(term_frequencies[word]) / num_words
    return term_frequencies
# End of term_frequency()


def inverse_document_frequency(lemmatized_records, unique_words):
    """Calculates the inverse document frequency of the tokenized lemmas in the records. Inverse document frequency
    is calculated as log (number of records in which word appears / total number of records)

    Params:
    - lemmatized_records (list): The lemmatized versions of the records loaded from the JSON file
    - unique_words (list): The list of unique words in the records

    Returns:
    - inverse_document_frequencies (dict): The inverse document frequency of each word
    """
    inverse_document_frequencies = dict()
    num_records = float(len(lemmatized_records))
    for word in unique_words:
        num_records_present = 0.0
        for record in lemmatized_records:
            if word in record:
                num_records_present += 1
        idf = math.log(num_records / num_records_present, 10)
        inverse_document_frequencies[word] = idf
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
    contents = map(lambda record: record[constants.TEXT].encode('utf-8'), records)
    word_tokenized_records = [wordcount.word_tokenize(record.lower()) for record in contents]
    lemmatized_records = wordcount.lemmatize_records(word_tokenized_records)
    tokenized_records = list()
    for lemmatized_record in lemmatized_records:
        tokenized_records.extend(lemmatized_record)
    term_frequencies = term_frequency(tokenized_records)
    inverse_document_frequencies = inverse_document_frequency(lemmatized_records, term_frequencies.keys())
    tf_idf_scores = list()
    for word in term_frequencies.keys():
        tf_idf_scores.append((word, term_frequencies[word] * inverse_document_frequencies[word]))
    return tf_idf_scores
# End of tf_idf()


def extract_important_words(tf_idf_scores, num_words):
    """Extracts important words based on their tf_idf_scores.

    Params:
    - tf_idf_scores (list): The tf-idf score of each word in the document
    - num_words (int): The number of words to extract

    Returns:
    - important_words (list): The list of words with the highest tf-idf score
    """
    sorted_scores = sorted(tf_idf_scores, key=lambda score: score[1], reverse=True)
    important_words = [score[0] for score in sorted_scores[:num_words]]
    return important_words
# End of extract_important_words()


if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file, False)
    tf_idf_scores = tf_idf(records)
    important_words = extract_important_words(tf_idf_scores, args.num_words)
    print(important_words)
