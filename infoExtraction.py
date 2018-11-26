import wordcount
import constants
import nltk
import nltk.corpus
import spacy
import numpy
from functools import reduce
import re
import time  # TODO remove
from collections import Counter

nlp = spacy.load('en_core_web_sm')


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
    So through all sentences in parsed and extract regex matchings, return the most frequent of these
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
        counted_matches = Counter(regex_matches)
        return counted_matches.most_common(1)[0][0]
    else:
        return '___no_match___'
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


def extract_wind_information(quantities):
    """
    Extract a high and low speed for wind from the given sentences
    :param quantities:
    :return: A tuple of the form (low_value, high_value, mean)
    """
    # From quantities filter to just sentences that have wind or winds in them
    wind_sents = filter_to_relevant_sentences(['wind', 'winds'], quantities)

    # Collect units from these sentences that are in terms of speed (put in speeds list)
    speed_units = ['mph', 'miles per hour', 'kph', 'kilometers per hour']
    speeds = extract_spacy_tag(wind_sents, 'QUANTITY')
    correct_unit_speeds = list(filter(lambda s: any(word in s for word in speed_units), speeds))
    converted_speeds = list(map(lambda s: convert_to_mph(s), correct_unit_speeds))

    # Throw out low numbers as even category 1 hurricanes have winds more than 70 mph
    higher_speeds = list(filter(lambda x: x > 40, converted_speeds))
    wind_speed_mean = numpy.mean(higher_speeds)
    wind_speed_std = numpy.std(higher_speeds)
    return int(wind_speed_mean - wind_speed_std), int(wind_speed_mean + wind_speed_std), int(wind_speed_mean)
# End of extract_wind_information


def extract_rain_information(quantities):
    """
    Extract all rain measurements from given sentences and return a range of high and low values for
    rain during the hurricane
    :param quantities:
    :return: A tuple of the form (low_value, high_value, mean)
    """
    # From quantities filter to just sentences that have wind or winds in them
    rain_sents = filter_to_relevant_sentences(['rain'], quantities)

    # Collect units from these sentences that are in terms of speed (put in speeds list)
    measurement_units = ['inches', 'in', 'centimeters', 'cm']
    measurements = extract_spacy_tag(rain_sents, 'QUANTITY')
    correct_unit_measurements = list(filter(lambda s: any(word in s for word in measurement_units), measurements))
    converted_measurements = list(map(lambda s: convert_to_in(s), correct_unit_measurements))

    rain_mean = numpy.mean(converted_measurements)
    rain_std = numpy.std(converted_measurements)
    return int(rain_mean - rain_std), int(rain_mean + rain_std), int(rain_mean)
# End of extract_rain_information


def extract_landfall_information(parsed):
    """
    Extract places the storm made landfall
    :param parsed:
    :return: The two most frequent matches as a tuple
    """
    landfall_words = ['landfall', 'approach']
    landfall_sents = filter_to_relevant_sentences(landfall_words, parsed)

    landfall_areas = extract_spacy_tag(landfall_sents, 'GPE')
    landfall_dates = extract_spacy_tag(landfall_sents, 'DATE')

    avg_date = get_average_date(landfall_dates)
    counted_areas = Counter(landfall_areas)
    landfall_cat = extract_frequent_regex_match(landfall_sents, '[Cc]ategory ([0-9]+)')

    # todo change to only get most frequent to be more general, also update the sentence
    most_common = counted_areas.most_common(2)
    return most_common[0][0], most_common[1][0], landfall_cat, avg_date
# End of extract_landfall_information


def extract_formation_date(dates, hurricane_name):
    """
    Given a list of sentences, use them to find the date that the tropical storm formed
    :param dates: List of sentences tagged
    :return: The formation date
    """
    formation_words = ['formed', 'Formed']
    formation_date_sents = filter_to_relevant_sentences(formation_words, dates)
    formation_dates = extract_spacy_tag(formation_date_sents, 'DATE')
    return formation_date_sents
# End of extract_formation_date


def extract_information(preprocessed_sentences):
    """
    Takes in an array of tokenized and pos tagged sentences and extracts information from them
    :param preprocessed_sentences: An array of paragraphs that have been processed
    :return: A large array of all of the named entities in no order
    """
    print("--- Starting nlp tagging: %s seconds ---" % (time.time() - start_time))
    parsed = list(map(lambda sentence: nlp(sentence), preprocessed_sentences))

    print("--- Starting filtering to quantities: %s seconds ---" % (time.time() - start_time))
    quantities = list(filter(lambda sentence: sentence_has_type(sentence, 'QUANTITY'), parsed))
    dates = list(filter(lambda sentence: sentence_has_type(sentence, 'DATE'), parsed))

    print("--- Starting hurricane name extraction: %s seconds ---" % (time.time() - start_time))
    hurricane_name = extract_frequent_regex_match(parsed, '[Hh]urricane ([A-Z][a-z]+)')

    print("--- Starting hurricane category extraction: %s seconds ---" % (time.time() - start_time))
    hurricane_category = extract_frequent_regex_match(parsed, '[Cc]ategory ([0-9]+)')

    print("--- Starting hurricane landfall info extraction: %s seconds ---" % (time.time() - start_time))
    landfall_info = extract_landfall_information(parsed)

    print("--- Starting wind extraction: %s seconds ---" % (time.time() - start_time))
    wind_range = extract_wind_information(quantities)

    print("--- Starting rain extraction: %s seconds ---" % (time.time() - start_time))
    rain_range = extract_rain_information(quantities)

    print(constants.HURRICANE_SENTENCE.format(hurricane_name, hurricane_category))
    print(constants.LANDFALL_SENTENCE.format(hurricane_name, landfall_info[2], landfall_info[3], landfall_info[0],
                                             landfall_info[1]))
    print(constants.WIND_SENTENCE.format(wind_range[0], wind_range[1], wind_range[2]))
    print(constants.RAIN_SENTENCE.format(hurricane_name, rain_range[1], rain_range[0], rain_range[2]))

    # print("--- Starting formation extraction: %s seconds ---" % (time.time() - start_time))
    # formation_date = extract_formation_date(dates, hurricane_name)
    # print(formation_date)
# End of extract_information


if __name__ == "__main__":
    start_time = time.time()
    # Load arguments and then the records from file
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file)

    # From the file, extract just the sentences_t sections and keep them as a list
    article_texts = list(map(lambda record: record[constants.TEXT], records))

    # Pre-process
    print("--- Starting preprocessing: %s seconds ---" % (time.time() - start_time))
    processed_texts = preprocess(article_texts)

    # Extract the named entities
    print("--- Starting extraction: %s seconds ---" % (time.time() - start_time))
    extract_information(processed_texts)
# End of main
