# coding: utf-8
"""Constructs bag-of-words-inspired feature sets out of the preprocessed records.
"""

import wordcount
import tfidf
import synsets


def get_bag_of_words_labels(preprocessed_records, args):
    """Gets the labels for the bag of words. A label can be a a single important word, a collocation of two important
    words, or a set of synonyms of a word.

    Params:
    - preprocessed_records (pyspark.rdd.RDD): The tokenized, lemmatized, lowercase records
    - ars (argparse.Namespace): The command-line arguments passed to the program

    Returns:
    - bag_of_words_labels (list<str|tuple<str>>): The labels of the bag of words created
    """
    frequent_words = wordcount.extract_frequent_words(preprocessed_records, args.num_words * 10, False)
    frequent_words = dict(frequent_words)
    frequent_collocations = wordcount.extract_collocations(preprocessed_records, args.num_collocations,
                                                           args.collocation_window)
    tf_idf_scores = tfidf.tf_idf(preprocessed_records)
    # Pyspark technically ends here - the rest is processed on master node
    important_words = tfidf.extract_important_words(tf_idf_scores, args.num_words, True)
    important_words_with_counts = synsets.add_word_counts(important_words, frequent_words)
    synset_dict = synsets.generate_syn_set(important_words_with_counts.items())
    words_and_collocations = wordcount.merge_collocations_with_wordlist(frequent_collocations,
                                                                        important_words_with_counts.items())
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
            if count > 0.0:
                print("Found ", label, " in ", words_in_record)
        elif " " in label:  # label represents a bigram
            for bigram in bigrams_in_record:
                if label == bigram:
                    count += 1.0
            if count > 0.0:
                print("Found ", label, " in ", bigrams_in_record)
        else:  # label represents an important word
            for word in words_in_record:
                if word == label:
                    count += 1.0
            if count > 0.0:
                print("Found ", label, " in ", words_in_record)
        feature_set.append(count)
    return (index, feature_set)
# End of make_count_feature_set()


if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file, False)
    preprocessed_records = wordcount.preprocess_records(records)
    # print("count of records = ", records.count())
    # print("count of preprocessed records = ", preprocessed_records.count())
    bag_of_words_labels = get_bag_of_words_labels(preprocessed_records, args)
    presence_feature_set = preprocessed_records.map(
        lambda record: make_presence_feature_set(record, bag_of_words_labels))
    wordcount.rdd_show(records, "=====Records=====")
    wordcount.rdd_show(preprocessed_records, "=====Preprocessed Records=====")
    wordcount.rdd_show(presence_feature_set, "=====Presence Feature Set=====")
    count_feature_set = preprocessed_records.map(lambda record: make_count_feature_set(record, bag_of_words_labels))
    wordcount.rdd_show(count_feature_set, "=====Count Feature Set=====")
    print("=====Bag of Words Labels=====")
    print(bag_of_words_labels)
