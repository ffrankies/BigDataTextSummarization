# Unit 1, read json file, and change it to the wanted format
import json
import re
from collections import Counter

def get_records_from_file(file_name):

    with open(file_name) as input_f:
        lines = input_f.readlines()

    records = []
    for record in lines:
        records.append(json.loads(record))

    return records


def process_words(input_data):
    word_list = []
    for line in input_data:
        #print(line['Sentences_t'].lower())
        words_lower = (line['Sentences_t'].lower())
        #word_list.append(re.findall(r'\w+',words_lower))

        for word in re.findall(r'\w+',words_lower):
            word_list.append(word)

    return Counter(word_list)


def start():
    records = get_records_from_file("test.json")
    muw_list = process_words(records)
    print(muw_list)

start()
