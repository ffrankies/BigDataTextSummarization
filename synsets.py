# coding: utf-8
from nltk.corpus import wordnet

import wordcount
import tfidf

import constants



def generate_syn_set(freq_list_complete):
    """
    Takes in a list of words and their frequencies and returns a dictionary with each word mapped
    to the frequency of it and its synonyms ex. "firing : [1961, 'attack', 'fire', 'shooting', 'make']"
    :param freq_list_complete: The output of running FreqDist() on a processed list
    :return: A dictionary of the above form
    """
    # Get a set of just the words from a tuple list
    freq_hash = set(map(lambda tup: tup[0], freq_list_complete))

    # Where to hold all synonyms of the words
    general_synonyms = generate_related_set(freq_hash)

    # Where to store the words with their synonyms that actually appear in the freq list
    relevant_synonyms = {}

    # Store words in relevant_synonyms mapped to an empty array
    for curr_syn in general_synonyms:
        relevant_synonyms[curr_syn] = [0]

    for word in general_synonyms:
        curr_syns = general_synonyms[word]
        for syn in curr_syns:
            if syn in freq_hash and \
                    syn not in relevant_synonyms[word]:
                relevant_synonyms[word].append(syn)
                word_count = list(filter(lambda x: x[0] == syn, freq_list_complete))[0][1]
                relevant_synonyms[word][0] = relevant_synonyms[word][0] + word_count
    return relevant_synonyms
# End of generate_syn_set


def generate_related_set(freq_words):
    """
    Get the synonyms and their hypernyms/lemmas for each of the top 10 most frequent works and
    store the word mapped to them. The result will be returned
    :param freq_words: A list of words to find related words of
    :return: A Dictionary of provided words mapped to a list of related words.
    """
    general_syns = {}
    for curr_set in freq_words:
        curr_general_synonyms = wordnet.synsets(curr_set)
        extended_synonyms = []
        for curr_syn in curr_general_synonyms:
            extended_synonyms.extend(curr_syn.hypernyms())

        # Get just the english word from the synset string
        extended_syns = list(map(lambda x: x.name().split('.')[0], extended_synonyms))
        extended_syns.append(curr_set)
        general_syns.update({curr_set: extended_syns})
    return general_syns
# End of generate_related_set


def print_syn_set(syn_set, num):
    """
    Print the synset in a nicely formatted manner
    :param syn_set: The set to be printed (generated from the generate_syn_set method)
    :param num: the number of synsets to print
    """
    count = 0
    for w in sorted(syn_set, key=syn_set.get, reverse=True):
        if syn_set[w][0] != 0:
            print("\item %s (%d): " % (w, syn_set[w][0]), syn_set[w][1:])
            count += 1
        if count > num:
            break
# End of print_syn_set


def add_word_counts(wordlist, word_counts):
    """Add word counts to words in wordlist, return result as a dictionary mapping word to word count.

    :param wordlist: The list of words to which word counts will be added
    :param word_counts: The dictionary of word counts

    :return: A dictionary mapping words in wordlist to their word counts
    """
    wordlist_with_counts = dict()
    for word in wordlist:
        if word in word_counts:
            wordlist_with_counts[word] = word_counts[word]
        else:
            print(word, " not in word_counts dictionary but in given wordlist")
    return wordlist_with_counts
# End of add_word_counts()


if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file, False)
    records = wordcount.preprocess_records(records)
    frequent_words = wordcount.extract_frequent_words(records, args.num_words * 10, False)
    frequent_words = dict(frequent_words)
    tf_idf_scores = tfidf.tf_idf(records)
    # Pyspark technically ends here - the rest is processed on master node
    important_words = tfidf.extract_important_words(tf_idf_scores, args.num_words, True)
    important_words_with_counts = add_word_counts(important_words, frequent_words)
    synset_dict = generate_syn_set(important_words_with_counts.items())
    print_syn_set(synset_dict)

