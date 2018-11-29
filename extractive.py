from pyteaser import SummarizeUrl, Summarize
import json
import os
import re

stripe_dir = "../Hurricane_Florence_small"
files_in_dir = [entry for entry in os.listdir(stripe_dir)  if os.path.isfile(os.path.join(stripe_dir, entry))]
files_in_dir = [os.path.join(stripe_dir, entry) for entry in files_in_dir]
json_files = [entry for entry in files_in_dir if re.match(r'part-.*\.json', os.path.basename(entry))]

total_summaries = []

# Summarizing from URL 
for current_file in json_files:
	with open(current_file, 'r') as json_file:
		lines = json_file.readlines()
		json_objects = [json.loads(line) for line in lines]
		for dictionary in json_objects:
			url = dictionary['URL_s']
			summary_sentences = SummarizeUrl(url)
			if summary_sentences:  # If list is not empty or None
				total_summaries.append(" ".join(summary_sentences))
				#print(summary_sentences)

total_summaries1 = [" ".join(total_summaries[i:i+20]) for i in range(0,len(total_summaries),20)]
total_summaries2 = []
for x in range(0, len((total_summaries1))):
	current_summary = total_summaries1[x]
	current_summary = Summarize("Hurricane Florence",current_summary)
	total_summaries2.append(current_summary)

print("Printing total summaries 2")
print(total_summaries2)

#final_summary = [" ".join(total_summaries2[i:i+20]) for i in range(0,len(total_summaries2),20)]
final_summary = ""
total_summaries3 = []
for y in range(0,len(total_summaries2)):
	#final_summary.append(total_summaries2[y][0])
	#final_summary = final_summary+" "+total_summaries2[y][0]
	total_summaries3.append(total_summaries2[y][0])

total_summaries4 = []

for x in range(0, len((total_summaries3))):
	current_summary = total_summaries3[x]
	current_summary = Summarize("Hurricane Florence",current_summary)
	total_summaries4.append(current_summary)

print("Printing total summaries 4")
print(total_summaries4)
	
print(final_summary)

for y in range(0,len(total_summaries2)):
	#final_summary.append(total_summaries2[y][0])
	final_summary = final_summary+" "+total_summaries4[y][0]
	

