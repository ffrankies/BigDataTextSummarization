from nltk import pos_tag
import wordcount
import constants
from nltk.tokenize import word_tokenize
import tfidf
import synsets

'''
Takes in tokenized records and tags words according to part of speech. Groups all the noun/verb tags we're looking for and 
throws words that are tagged as such into respective lists. Can be modified to use case later. 
Gets the most frequent words and groups them into synsets. 
Noun synsets seem alright, verb synsets aren't that great. TODO 
'''
def pos_tagging(records): 
	#pos tag records 
	tagged_records = map(lambda record: pos_tag(record), records)
	tagged_records = wordcount.filter_stopwords(tagged_records)

	noun_pos = ["NN", "NNS", "NNP", "NNPS"]
	verb_pos = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]	
	nouns = []
	verbs = []

	for article in tagged_records:
		[nouns.append(token[0]) if token[1] in noun_pos\
		 else verbs.append(token[0]) if token[1] in verb_pos else '' for token in article]
	
	frq_nouns = wordcount.extract_frequent_words(nouns, 500)
	frq_verbs = wordcount.extract_frequent_words(verbs, 500)

	synset_nouns = synsets.generate_syn_set(frq_nouns)
	synset_verbs = synsets.generate_syn_set(frq_verbs)

	print("Synset--NOUNS")
	synsets.print_syn_set(synset_nouns)
	print("##################")
	print("Synset--VERBS")
	synsets.print_syn_set(synset_verbs)

if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file)
    tf_idf_scores = tfidf.tf_idf(records)
    important_words = tfidf.extract_important_words(tf_idf_scores, args.num_words)
    contents = map(lambda record: record[constants.TEXT], records) #tokenize records 
    tokenized_records = [word_tokenize(record.lower()) for record in contents]
    print(tokenized_records)
    #pos_tagging(tokenized_records)
