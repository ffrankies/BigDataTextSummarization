import wordcount
import constants
import nltk.corpus
from collections import Counter


def preprocess(raw_texts):
    """
    Takes in a list of paragraph texts and tokenizes then pos tags them
    :param raw_texts: list of paragraphs from the articles
    :return: the same list after being pos tagged and tokenized
    """
    # Tokenize
    raw_texts = list(map(lambda x: nltk.word_tokenize(x), raw_texts))
    # Pos Tag
    raw_texts = list(map(lambda x: nltk.pos_tag(x), raw_texts))
    return raw_texts
# End of preprocess


def extract_named_entities(preprocessed_texts):
    """
    Takes in an array of tokenized and pos tagged sentences and extracts the named entities from them
    :param preprocessed_texts: An array of paragraphs that have been processed
    :return: A large array of all of the named entities in no order
    """
    # Chunk all named entities in the texts
    pos_tagged = list(map(lambda text: nltk.ne_chunk(text), preprocessed_texts))

    # Flatten the Array of arrays and then remove all tuples (since named entities will be in a tree form right now)
    flat_pos_tagged = [item for sublist in pos_tagged for item in sublist]
    pos_tagged_trees = list(filter(lambda word: not isinstance(word, tuple), flat_pos_tagged))

    # Get just the leaves of the trees and return them as a list of strings
    tree_leaves = list(map(lambda tree: tree.leaves(), pos_tagged_trees))
    return list(map(lambda arr: arr_to_string(arr), tree_leaves))
# End of extract_named_entities


def arr_to_string(pos_array):
    """
    Takes a (word, pos) array and converts it into a string of just the words
    :param pos_array:
    :return: The string after the conversion
    """
    result_string = ''
    for pos in pos_array:
        result_string += pos[0]
        result_string += ' '

    # Remove trailing space
    result_string = result_string[:-1]
    return result_string
# End of arr_to_string


if __name__ == "__main__":
    # Load arguments and then the records from file
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file)

    # From the file, extract just the sentences_t sections and keep them as a list
    article_texts = list(map(lambda record: record[constants.TEXT], records))

    # Pre-process
    processed_texts = preprocess(article_texts)

    # Extract the named entities
    named_entities = extract_named_entities(processed_texts)

    # Count and print the occurrences
    counted_named_entities = Counter(named_entities)

    for entity in counted_named_entities.most_common(50):
        print('\item %s : %d' % (entity[0], entity[1]))
