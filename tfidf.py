"""Uses tf-idf to extract important words from a collection of documents.
"""

import math

from nltk.probability import FreqDist

import wordcount


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
    num_words = len(tokenized_records)
    for word in unique_words:
        term_frequencies[word] = term_frequencies[word] / num_words
    return term_frequencies
# End of term_frequency()


def inverse_document_frequency(records, unique_words):
    """Calculates the inverse document frequency of the tokenized lemmas in the records. Inverse document frequency
    is calculated as log (number of records in which word appears / total number of records)

    Params:
    - records (list): The records loaded from the JSON file
    - unique_words (list): The list of unique words in the records

    Returns:
    - inverse_document_frequencies (dict): The inverse document frequency of each word
    """
    contents = map(lambda record: record[constants.TEXT], records)
    tokenized_records = [wordcount.word_tokenize(record.lower()) for record in contents]
    lemmatized_records = wordcount.lemmatize_words(tokenized_records)
    inverse_document_frequencies = dict()
    num_records = len(lemmatized_records)
    for word in unique_words:
        num_records_present = 0
        for record in lemmatized_records:
            if word in record:
                num_records_present += 1
        idf = math.log(num_records_present / num_records, 10)
        inverse_document_frequencies[word] = idf
    return inverse_document_frequencies
# End of inverse_document_frequency()


if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file, False)
    tokenized_records = wordcount.tokenize_records(records)
    term_frequency(tokenized_records)
