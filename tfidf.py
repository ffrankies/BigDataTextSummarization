"""Uses tf-idf to extract important words from a collection of documents.
"""

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
    word_counts = dict(word_counts)
    unique_words = word_counts.keys()
    num_words = len(tokenized_records)
    for word in unique_words:
        word_counts[word] = word_counts[word] / num_words
    return word_counts
# End of term_frequency()


if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file, False)
    tokenized_records = wordcount.tokenize_records(records)
    term_frequency(tokenized_records)
