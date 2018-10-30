import wordcount
import constants
import nltk
import nltk.corpus
import spacy
from functools import reduce
from spacy import displacy
from collections import Counter
nlp = spacy.load('en_core_web_sm')

NP_PATTERN = 'NP: {<DT>?<JJ>*<NN>}'

def preprocess(raw_texts):
    """
    Takes in a list of paragraph texts and tokenizes then pos tags them
    :param raw_texts: list of paragraphs from the articles
    :return: the same list after being pos tagged and tokenized
    """
    # Tokenize into sentences and then flat map to a list of sentences
    raw_texts = map(lambda x: nltk.sent_tokenize(x), raw_texts)
    raw_texts = reduce(list.__add__, raw_texts)
    return raw_texts
# End of preprocess


def extract_information(preprocessed_texts):
    """
    Takes in an array of tokenized and pos tagged sentences and extracts the named entities from them
    :param preprocessed_texts: An array of paragraphs that have been processed
    :return: A large array of all of the named entities in no order
    """
    # Chunk all named entities in the texts
    # return list(map(lambda text: nltk.ne_chunk(text), preprocessed_texts))
    # regex_parser = nltk.RegexpParser(NP_PATTERN)

    parsed = list(map(lambda x: list(map(lambda y: nlp(y), x)), preprocessed_texts))
    return parsed
# End of extract_named_entities


if __name__ == "__main__":
    # Load arguments and then the records from file
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file)

    # From the file, extract just the sentences_t sections and keep them as a list
    article_texts = list(map(lambda record: record[constants.TEXT], records))

    # Pre-process
    processed_texts = preprocess(article_texts)

    # Extract the named entities
    information = extract_information(processed_texts)
    # print([(X.text, X.label_) for X in doc.ents])
