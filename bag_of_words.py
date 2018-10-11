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
    - bag_of_words_labels (list<str|tuple<str>>): 
    """
    frequent_words = wordcount.extract_frequent_words(records, args.num_words * 10, False)
    frequent_words = dict(frequent_words)
    frequent_collocations = wordcount.extract_collocations(preprocessed_records, args.num_collocations,
                                                           args.collocation_window)
    tf_idf_scores = tfidf.tf_idf(records)
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


if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file, False)
    records = wordcount.preprocess_records(records)
    # frequent_words = wordcount.extract_frequent_words(records, args.num_words * 10, False)
    # frequent_words = dict(frequent_words)
    # tf_idf_scores = tfidf.tf_idf(records)
    # # Pyspark technically ends here - the rest is processed on master node
    # important_words = tfidf.extract_important_words(tf_idf_scores, args.num_words, True)
    # important_words_with_counts = synsets.add_word_counts(important_words, frequent_words)
    # synset_dict = synsets.generate_syn_set(important_words_with_counts.items())
    bag_of_words_labels = get_bag_of_words_labels(records, args)
    print("=====Bag of Words Labels=====")
    print(bag_of_words_labels)
