import re
from collections import Counter
from functools import reduce

import nltk
import nltk.corpus

import constants


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


def sentence_has_type(sentence, type):
    """
    Helper method that takes in a sentence and the current spacy entity type, and returns
    a true if that type is in the given sentence (used for filtering)
    :param sentence: Sentence to search through
    :param type: The spacy entity type to search for
    :return: boolean
    """
    for word in sentence.ents:
        if word .label_ == type:
            return True
    return False
# End of sentence_has_type


def filter_to_relevant_sentences(relevant_words, sentences):
    return list(filter(lambda quantity_sent: any(word in quantity_sent.text for word in relevant_words), sentences))
# End of filter_to_relevant_sentences


def convert_to_mph(text):
    """
    Given a string with some numbers and their speed units, extract the numbers and convert units appropriatly
    :param text: The string to convert
    :return: One number representing the average number in the string (if more than one) in mph
    """
    # Get all numbers from the string and cast them to ints and average them
    nums = re.findall('([0-9]+)', text)
    average = 0
    if len(nums) > 1:
        average = reduce(lambda x, y: int(int(x) + int(y)), nums) / len(nums)
    elif nums:
        average = int(nums[0])
    else:
        average = 0

    mph = ['mph', 'miles per hour']
    kph = ['kph', 'kilometers per hour']

    if any(word in text for word in mph):
        return average
    elif any(word in text for word in kph):
        # Convert to mph and return
        return average * .62137
    else:
        return -1
# End of convert_to_mph


def convert_to_in(text):
    """
    Given a string with some numbers and their speed units, extract the numbers and convert units appropriatly
    :param text: The string to convert
    :return: One number representing the average number in the string (if more than one) in mph
    """
    # Get all numbers from the string and cast them to ints and average them
    nums = re.findall('([0-9]+)', text)
    average = 0
    if len(nums) > 1:
        average = reduce(lambda x, y: int(int(x) + int(y)), nums) / len(nums)
    elif nums:
        average = int(nums[0])
    else:
        average = 0

    inches = ['in', 'inches']
    cm = ['cm', 'centimeters']

    if any(word in text for word in inches):
        return average
    elif any(word in text for word in cm):
        # Convert to mph and return
        return average / 2.54
    else:
        return -1
# End of convert_to_mph


def extract_frequent_regex_match(parsed, regex):
    """
    Go through all sentences in parsed and extract regex matchings, return the most frequent of these
    :param parsed: spacy tagged sentences
    :param regex: regex expression to search for
    :return: the most frequent regex match
    """
    regex_matches = []

    for sentence in parsed:
        matches = re.findall(regex, sentence.text)
        if matches:
            regex_matches.extend(matches)

    if regex_matches:
        return Counter(regex_matches)
    else:
        return '___no_match___'
# End of extract_frequent_regex_match


def filter_regex_match_sentences(parsed, pattern):
    """
    filter parsed to only contain sentences with a matching regex form
    :param parsed: spacy tagged sentences
    :param regex: regex expression to search for
    :return: the filtered list
    """
    matches = list(filter(lambda sent: re.findall(pattern, sent.text), parsed))
    return matches
# End of extract_frequent_regex_match


def get_average_date(date_list):
    """
    Given a list of dates, extract the average date given
    :param date_list: A list of dates in varying formats
    :return: The most frequent date
    """
    month_count = [0] * 12
    month_dates = [[], [], [], [], [], [], [], [], [], [], [], []]

    # Count frequency of each month, and sort dates by their month
    for date in date_list:
        for i in range(12):
            if constants.MONTH_NAMES[i] in date:
                month_count[i] += 1
                month_dates[i].append(date)

    # Find max count and get the sentences from that month
    max_count = -1
    most_freq_month = -1
    for j in range(12):
        if month_count[j] > max_count:
            max_count = month_count[j]
            most_freq_month = j
    freq_month_dates = month_dates[most_freq_month]
    freq_month = constants.MONTH_FULL_NAMES[most_freq_month]

    years = []
    days = []
    for date in freq_month_dates:
        nums = re.findall('([0-9]+)', date)
        for num in nums:
            if int(num) > 1900:
                years.append(num)
            elif int(num) < 31:
                days.append(num)

    counted_days = Counter(days)
    counted_years = Counter(years)

    return freq_month + ' ' + counted_days.most_common(1)[0][0] + ', ' + counted_years.most_common(1)[0][0]
# End of get_average_date


def extract_spacy_tag(sentences, tag):
    """
    Helper for other extraction methods. Takes a list of spacy tagged sentences
    and gets all words of a certain type from them
    :param sentences:
    :param tag:
    :return:
    """
    tagged = []
    for sent in sentences:
        for entity in sent.ents:
            if entity.label_ == tag:
                # Some entries had hyphens and parentheses in them, so this removes them
                clean_tagged = re.sub('[^a-zA-Z0-9]', ' ', entity.text)
                tagged.append(clean_tagged)
    return tagged
# End of extract_spacy_tag