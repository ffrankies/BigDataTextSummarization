"""Uses Pyteaser to create an extractive summary
"""
import json
import os
import re
import argparse

from pyteaser import SummarizeUrl, Summarize


def parse_arguments():
    """Parses command-line arguments.

    Returns:
    - args (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='The path to the directory containing the dataset')
    return parser.parse_args()
# End of parse_arguments()


def get_json_files(directory):
    """Extracts all the JSON files from a directory.

    Params:
    - directory (str): The directory containing JSON files

    Returns:
    - json_files (list<str>): The list of paths to the JSON files
    """
    stripe_dir = directory
    files_in_dir = [entry for entry in os.listdir(stripe_dir)  if os.path.isfile(os.path.join(stripe_dir, entry))]
    files_in_dir = [os.path.join(stripe_dir, entry) for entry in files_in_dir]
    json_files = [entry for entry in files_in_dir if re.match(r'part-.*\.json', os.path.basename(entry))]
    return json_files
# End of get_json_files()

if __name__ == "__main__":
    args = parse_arguments()
    json_files = get_json_files(args.dataset)
    total_summaries = []

    # Summarizing from URL 
    print("Summarizing from URLs")
    for current_file in [json_files[0]]:
        with open(current_file, 'r') as json_file:
            for line in json_file:
                record = json.loads(line)
                url = record['URL_s']
                print("Summarizing...", url)
                summary_sentences = SummarizeUrl(url)
                if summary_sentences:
                    total_summaries.append(" ".join(summary_sentences))
        print("Done processing one file")

    print("Finished first pass through all records")
    print("Recombining and summarizing...")
    while len(total_summaries) > 15:
        summaries_to_join = int(len(total_summaries) / 15)
        if summaries_to_join == 1:
            break
        if summaries_to_join > 20:
            summaries_to_join = 20
        combined_summaries = [" ".join(total_summaries[i:i+summaries_to_join]) 
                              for i in range(0, len(total_summaries), summaries_to_join)]
        total_summaries = [" ".join(Summarize("Hurricane Florence", summary).split("\n")) 
                           for summary in combined_summaries]
        print("Finished pass through recombined summaries... Number of summaries left = %d" % len(total_summaries))

    print("Final summary:")
    for summary in total_summaries:
        print(summary)
        print("\n")
