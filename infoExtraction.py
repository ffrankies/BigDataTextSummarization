import re
from collections import Counter

import numpy
import spacy
import word2number as w2n

import constants
import extractionHelpers as eh
import wordcount

nlp = spacy.load('en_core_web_sm')

def extract_wind_information(quantities):
    """
    Extract a high and low speed for wind from the given sentences
    :param quantities:
    :return: A tuple of the form (low_value, high_value, mean)
    """
    # From quantities filter to just sentences that have wind or winds in them
    wind_sents = eh.filter_to_relevant_sentences(['wind', 'winds'], quantities)

    # Collect units from these sentences that are in terms of speed (put in speeds list)
    speed_units = ['mph', 'miles per hour', 'kph', 'kilometers per hour']
    speeds = eh.extract_spacy_tag(wind_sents, 'QUANTITY')
    correct_unit_speeds = list(filter(lambda s: any(word in s for word in speed_units), speeds))
    converted_speeds = list(map(lambda s: eh.convert_to_mph(s), correct_unit_speeds))

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
    rain_sents = eh.filter_to_relevant_sentences(['rain'], quantities)

    # Collect units from these sentences that are in terms of speed (put in speeds list)
    measurement_units = ['inches', 'in', 'centimeters', 'cm']
    measurements = eh.extract_spacy_tag(rain_sents, 'QUANTITY')
    correct_unit_measurements = list(filter(lambda s: any(word in s for word in measurement_units), measurements))
    converted_measurements = list(map(lambda s: eh.convert_to_in(s), correct_unit_measurements))

    rain_mean = numpy.mean(converted_measurements)
    rain_std = numpy.std(converted_measurements)
    return int(rain_mean - rain_std), int(rain_mean + rain_std), int(rain_mean)
# End of extract_rain_information


def extract_size_information(quantities):
    size_sentences = eh.filter_to_relevant_sentences(['diameter', 'radius'], quantities)
    size_quantities = eh.extract_spacy_tag(size_sentences, 'QUANTITY')

    miles = []
    for quantity in size_quantities:
        matches = re.findall('([0-9]+) miles', quantity)
        if matches:
            miles.append(int(matches[0]))
    return numpy.mean(miles)
# End of extract_size_information


def extract_landfall_information(parsed):
    """
    Extract info about places the storm made landfall
    :param parsed:
    :return: The two most frequent matches as a tuple
    """
    landfall_words = ['landfall', 'approach']
    landfall_sents = eh.filter_to_relevant_sentences(landfall_words, parsed)

    landfall_areas = eh.extract_spacy_tag(landfall_sents, 'GPE')
    landfall_dates = eh.extract_spacy_tag(landfall_sents, 'DATE')

    avg_date = eh.get_average_date(landfall_dates)
    counted_areas = Counter(landfall_areas)
    landfall_cat = eh.extract_frequent_regex_match(landfall_sents, '[Cc]ategory ([0-9]+)').most_common(1)[0][0]

    # todo change to only get most frequent to be more general, also update the sentence
    most_common = counted_areas.most_common(2)
    return most_common[0][0], most_common[1][0], landfall_cat, avg_date
# End of extract_landfall_information


# def extract_formation_info(parsed):
#     """
#     Given a list of sentences, use them to find the date that the tropical storm formed
#     :param dates: List of sentences tagged
#     :return: The formation date
#     """
#     formation_sents = filter_regex_match_sentences(parsed, '([Hh]urricane [a-zA-z\s]* [was\s]*formed)')
#     formation_dates = extract_spacy_tag(formation_sents, 'DATE')
#     formation_gpes = extract_spacy_tag(formation_sents, 'GPE')
#
#     return True
# # End of extract_formation_info


def extract_death_damages_info(quantities):
    """
    Extract information related to hurricane deaths
    :param quantities: sentences
    :return: the information
    """
    death_toll = []
    death_sentences = eh.filter_to_relevant_sentences(['deaths', 'death'], quantities)
    for sent in death_sentences:
        death_nums = re.findall('([0-9]+) deaths', sent.text)
        death_toll.extend(death_nums)

        death_words = re.findall('([a-z]+) deaths', sent.text)
        for word in death_words:
            try:
                number = w2n.word_to_num(word)
                death_toll.append(number)
            except:
                pass

    damages_sentences = eh.filter_to_relevant_sentences(['damage', 'damages'], quantities)
    damages_quantities = eh.extract_spacy_tag(damages_sentences, 'QUANTITY')

    injuries_sentences = eh.filter_to_relevant_sentences(['injured', 'injuries'], quantities)
    injury_quantities = eh.extract_spacy_tag(injuries_sentences, 'QUANTITY')

    return True
# End of extract_death_info


def extract_preparation_information(parsed):
    evacuation_sents = eh.filter_to_relevant_sentences(['evacuate', 'evacuated', 'evacuation'], parsed)
    evacuation_quantities = eh.extract_spacy_tag(evacuation_sents, 'QUANTITY')
    evacuation_gpe = eh.extract_spacy_tag(evacuation_sents, 'GPE')

    people_evacuated = []
    for sent in evacuation_sents:
        for entity in sent.ents:
            try:
                next_words = re.findall(entity.text + ' ([a-zA-z]+)', sent.text)
            except:
                continue

            if next_words and next_words[0] == 'people':
                people_evacuated.append(entity.text)

    counted_gpes = Counter(evacuation_gpe)
    counted_people = Counter(people_evacuated)

    evacuated_numbers = []
    for people in people_evacuated:
        matches = re.findall('([0-9,]+.*)', people)
        if matches:
            evacuated_numbers.append(matches[0])

    counted_evacuated = Counter(evacuated_numbers)
    return counted_gpes, counted_evacuated
# End of extract_preparation_information


def extract_restoration_information(parsed):
    restoration_sents = eh.filter_to_relevant_sentences(['restored', 'restoration', 'aid', 'shelter'], parsed)

    restoration_quantities = eh.extract_spacy_tag(restoration_sents, 'QUANTITY')
    restoration_gpe = eh.extract_spacy_tag(restoration_sents, 'GPE')
    restoration_orgs = eh.extract_spacy_tag(restoration_sents, 'ORG')

    return True
# End of extract_restoration_information


def extract_information(preprocessed_sentences):
    """
    Takes in an array of tokenized and pos tagged sentences and extracts information from them
    :param preprocessed_sentences: An array of paragraphs that have been processed
    :return: A large array of all of the named entities in no order
    """
    parsed = list(map(lambda sentence: nlp(sentence), preprocessed_sentences))

    quantities = list(filter(lambda sentence: eh.sentence_has_type(sentence, 'QUANTITY'), parsed))
    dates = list(filter(lambda sentence: eh.sentence_has_type(sentence, 'DATE'), parsed))

    hurricane_name = eh.extract_frequent_regex_match(parsed, '[Hh]urricane ([A-Z][a-z]+)').most_common(1)[0][0]
    hurricane_category = eh.extract_frequent_regex_match(parsed, '[Cc]ategory ([0-9]+)').most_common(1)[0][0]

    tropical_storm_name = eh.extract_frequent_regex_match(parsed, '[Tt]ropical [Ss]torm ([A-Z][a-z]+)').most_common(1)[0][0]
    preperation_info = extract_preparation_information(parsed)
    restore_info = extract_restoration_information(parsed)

    landfall_info = extract_landfall_information(parsed)

    wind_info = extract_wind_information(quantities)
    rain_info = extract_rain_information(quantities)
    size_info = extract_size_information(parsed)

    # formation_info = extract_formation_info(parsed)
    death_info = extract_death_damages_info(parsed)

    print(constants.HURRICANE_SENTENCE.format(hurricane_name, hurricane_category))
    print(constants.LANDFALL_SENTENCE.format(hurricane_name, landfall_info[2], landfall_info[3], landfall_info[0],
                                             landfall_info[1]))
    print(constants.WIND_SENTENCE.format(wind_info[0], wind_info[1], wind_info[2]))
    print(constants.RAIN_SENTENCE.format(hurricane_name, rain_info[1], rain_info[0], rain_info[2]))
    # print(formation_date)
# End of extract_information


if __name__ == "__main__":
    # Load arguments and then the records from file
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file)

    # From the file, extract just the sentences_t sections and keep them as a list
    article_texts = list(map(lambda record: record[constants.TEXT], records))

    # Pre-process
    processed_texts = eh.preprocess(article_texts)

    # Extract the named entities
    extract_information(processed_texts)
# End of main
