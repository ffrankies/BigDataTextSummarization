from nltk.corpus import wordnet


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


def print_syn_set(syn_set):
    """
    Print the synset in a nicely formatted manner
    :param syn_set: The set to be printed (generated from the generate_syn_set method)
    """
    for w in sorted(syn_set, key=syn_set.get, reverse=True):
        if syn_set[w][0] != 0:
            print("%15s"%w, syn_set[w])
# End of print_syn_set
