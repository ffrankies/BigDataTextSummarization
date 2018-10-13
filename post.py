from nltk import pos_tag
import wordcount
import constants
from nltk.tokenize import word_tokenize
import tfidf
import synsets
from nltk.stem import WordNetLemmatizer

NOUNS = ["NN", "NNS", "NNP", "NNPS"]
VERBS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
LEMMATIZER = WordNetLemmatizer()


def pos_parser(records):
	'''
	Tokenizes each article in the list of articles and tags each word according to part of speech and then filters out stopwords
	Params:
	-  records (list<str>): the contents of each record stored as a string in a list 
	Returns:
	- filtered_text (list[list[tuple<str,str>]): list of list of tuples. each sublist is a article containing tuples of <word, pos>
	for example given the 2 records:
		the dog slept.
		john is hungry. john made a sandwich 
	would return 
	[[('the', POS), ('dog', POS), ('slept', POS), ('.', '.')], [('john', POS) ...('sandwich', POS)]] 	
	'''
	tokenized_records = [word_tokenize(article.lower()) for article in records]
	text = list(map(lambda record: pos_tag(record), tokenized_records)) #tags every word into pos  
	filtered_text = wordcount.filter_stopwords(text) #filters out stopwords 	
	return filtered_text

def lemmatizer(wordlist): 
	'''
	Lemmatizes words in a list 
	Params:
	-  wordlist (list<str>):  a list of words to be lemmatized
	Return:
	- lemmatized_words (list<str>): a list of lemmatized words 
	'''
	lemmatized_words = []
	for w in wordlist:
		w = LEMMATIZER.lemmatize(w)
		lemmatized_words.append(w)
	return lemmatized_words

def noun_tagger(tagged_records): #tagged records is a list of lists 
	'''
	Looks for nouns in each individual record and throws them into a list 
	Params:
	- tagged_records (list[list[tuple<str,str>]): list of list of tuples. each sublist is a article containing tuples of <word, pos>
	Return:
	- noun_record (list[list<str>]): List of lists. Each sublist contains the tagged nouns in the record
	e.g. Given 3 records: 
		"The quick brown fox jumped over the lazy dog"
		"The phone is missing"
		"My car rolled down the hill" 
	would return 
	[["fox", "dog"], ["phone"], ["car", "hill"]]
	'''
	nouns = []
	noun_record = []

	for article in tagged_records:
		for token in article:
			if token[1] in NOUNS:
				nouns.append(token[0])
			noun_record.append(lemmatizer(nouns))
	return noun_record

def verb_tagger(tagged_records):
	'''
	Looks for verbs in each individual record and throws them into a list 
	Params:
	- tagged_records (list[list[tuple<str,str>]): list of list of tuples. each sublist is a article containing tuples of <word, pos>
	Return:
	- noun_record (list[list<str>]): List of lists. Each sublist contains the tagged verbs in the record
	'''
	verbs = []
	verb_record = []
	lemmatized_verbs = []
	for article in tagged_records:
		for token in article:
			if token[1] in VERBS:
				verbs.append(token[0])
			verb_record.append(lemmatizer(verbs))
	return verb_record


def pos_tfidf_scores(wordlist):
	'''
	Calculates the term frequency inverse document frequency for each lemmatized non-stopword in the dataset of
    records.
	Params:
	- wordlist (list[list<str>]): List of lists. Each sublist contains the tagged nouns in the record
	Return:
	-  tfidf_scores (list[tuple<str, double>]): list of tuples of <word, tfidf-score>.  
	'''
	tokenized_records = list()
	for lemmatized_record in wordlist:
		tokenized_records.extend(lemmatized_record)
	tf = tfidf.term_frequency(tokenized_records)
	idf = tfidf.inverse_document_frequency(wordlist, tf.keys())
	tfidf_scores = list()
	for word in tf.keys():
		tfidf_scores.append((word, tf[word]*idf[word]))
	return tfidf_scores


def pos_tag_nouns(records):
	'''
	Lemmatizes words in a list 
	Params:
	-  records (list<str>): the contents of each record stored as a string in a list i.e. a list of strings
	Return:
	- important_words (list<str>): a list of important nouns  
	'''
	tagged_records = pos_parser(records)
	nouns = noun_tagger(tagged_records)
	tfidf_scores = pos_tfidf_scores(nouns)
	important_words = tfidf.extract_important_words(tfidf_scores, len(nouns))
	return important_words

def pos_tag_verbs(records):
	'''
	
	Params:
	-  records (list<str>): the contents of each record stored as a string in a list i.e. a list of strings
	Return:
	- important_words (list): 	a list of important verbs  
	'''
	tagged_records = pos_parser(records)
	verbs = verb_tagger(tagged_records)
	tfidf_scores = pos_tfidf_scores(verbs)
	important_words = tfidf.extract_important_words(tfidf_scores, len(verbs))
	return important_words

if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file) #dictionary 
    contents = list(map(lambda record: record[constants.TEXT], records)) #puts records into a list from dictionary
    print("IMPORTANT NOUNS:\n")
    print (pos_tag_nouns(contents))
    print("IMPORTANT VERBS:\n")
    print (pos_tag_verbs(contents))
