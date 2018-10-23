# CS 4984/5984 Team 12

- [CS 4984/5984 Team 12](#cs-49845984-team-12)
    - [PySpark Instructions](#pyspark-instructions)
    - [What We're Doing](#what-were-doing)
        - [Unit 1 - A set of most frequent important words](#unit-1---a-set-of-most-frequent-important-words)
        - [Unit 2 - A set of WordNet synsets that cover the words](#unit-2---a-set-of-wordnet-synsets-that-cover-the-words)
        - [Unit 3 - A set of words constrained by POS, e.g., nouns and/or verbs](#unit-3---a-set-of-words-constrained-by-pos-eg-nouns-andor-verbs)
        - [Unit 4 - A set of words/word stems that are discriminating features (that also are helpful in a classifier for the relevant webpages)](#unit-4---a-set-of-wordsword-stems-that-are-discriminating-features-that-also-are-helpful-in-a-classifier-for-the-relevant-webpages)
        - [Unit 5 - A set of frequent & important named entities](#unit-5---a-set-of-frequent--important-named-entities)
        - [Unit 6 - A set of important topics, e.g., identified using LDA](#unit-6---a-set-of-important-topics-eg-identified-using-lda)
        - [Unit 7 - An extractive summary, as a set of important sentences, e.g., identified by clustering](#unit-7---an-extractive-summary-as-a-set-of-important-sentences-eg-identified-by-clustering)
        - [Unit 8 - A set of values for each slot matching collection semantics](#unit-8---a-set-of-values-for-each-slot-matching-collection-semantics)
        - [Unit 9 - A readable summary explaining the slots & values](#unit-9---a-readable-summary-explaining-the-slots--values)
        - [Unit 10 - A readable abstractive summary, e.g., from deep learning](#unit-10---a-readable-abstractive-summary-eg-from-deep-learning)
    - [Useful Info](#useful-info)
        - [Data](#data)
        - [Accounts](#accounts)
        - [Easy Log-In](#easy-log-in)
        - [Tools](#tools)
            - [Hadoop](#hadoop)
        - [Project Report](#project-report)

## PySpark Instructions

- Everyone should have a separate directory on the Hadoop cluster. Since they do not have git, if we make changes to one directory, we'll be messing with each other's work.
- I have an example set-up, which I've been using in `~/project/` (which I should probably rename).
- Example pyspark command: `spark2-submit --master=local[4] --driver-memory 5g --py-files=pyfiles.zip --properties-file=./spark-defaults.conf <yourscript>.py <yourscriptargs> 2> error.out > result.out`
- Here's what it does:
    - `--master=local[4]`: There are problems running scripts on the cluster, when they depend on other scripts we have written ourselves. This makes all scripts run on the head node (the TA came up with this work-around, though it's not ideal).
        - Can be run on cluster with `--master=yarn` instead, but it will have to be a standalone script with no dependencies.
    - `--driver-memory 5g`: Sets 5Gb of RAM aside for the head node, to avoid Java out-of-memory errors. My guess is we can ignore this.
    - `--py-files=pyfiles.zip`: A zip file containing all the python scripts that `<yourscript>.py` depends on. May also not be needed if we're running on the head node, but hey, don't fix what ain't broken.`
    - `--properties-file=./spark-defaults.conf`: I've created a configuration file with some extra options. Will be explained further out.
    - `2> error.out > result.out`: Spark does a ton of logging, which clutters the screen. This moves most Spark logging to error.out, and print statements from `<yourscript>.py` to result.out.
- Pyspark requires a zipped python environment to run external scripts. I have created one, called `nltk_env`, which at the moment should contain `nltk` and whatever anaconda comes with.
- To add an external library to the environment, you can use the `add_pyspark_package <yourpackages>` command, which I wrote and added to the bash shell. This will activate the environment, install the python package locally, copy the installed package to the conda environment directory, and then zip the conda environment into the `~/project/nltk_env.zip` file. You can then copy that zip file to your directory.
- The configuration file:
    - Here's an example:

    ```conf
    spark.yarn.appMasterEnv.PYSPARK_PYTHON=./NLTK/nltk_env/bin/python  # Used the zipped python environment
    spark.yarn.appMasterEnv.NLTK_DATA=./  # Tells Spark that NLTK data is in this directory (in the zip files)
    spark.executorEnv.NLTK_DATA=./  # Tells Spark that NLTK data is in this directory (in the zip files)
    spark.yarn.dist.archives=nltk_env.zip#NLTK,tokenizers.zip#tokenizers,taggers.zip#taggers,corpora.zip#corpora  # Upload zip files to the Spark manager
    spark.eventLog.enabled=true  # Add an event log
    spark.eventLog.dir=/home/cs4984cs5984f18_team12/logs/  # Save event log in home directory, under logs. Can be parsed with an external utility, if need be
    ```

## What We're Doing

### Unit 1 - A set of most frequent important words

- For unit one, we will be producing a 'summary' of our data by picking the most used words in our dataset
- This will require some pre-processing:
  - The data in the JSON file (`/sentences/part-0000-....-c000.json`) is not fully cleaned. There should still be some HTML artifacts that could be gotten rid of.
  - We will want to remove stop words and punctuation.
  - We may want to use the roots of words instead of the full words
- The `nltk` module has the `word_tokenize` function and the `FreqDist` class that should do a majority of the counting work for us. It also has a database of stop words somewhere.
- One way to do that: TF_IDF (Term Frequency - Inverse Document Frequency)
  - Words with highest (tf-idf) score are the 'most important'
- Another way (more experimental), is to look at word collocations (words used together)
  - The basic idea is, if two words are used together, we can use them either together (police + officer)
  - Or we can only pick one of them (police)
- According to [this dude](https://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/), another way is using graph theory, clustering or some machine learning black magic. [We didn't do this because it's hard]
  - Most time intensive, but most likely to give good results

### Unit 2 - A set of WordNet synsets that cover the words

- For unit two, we're building on top of unit 1
- We're looking at using synonyms of the words to give them more meaning / context
- For example, 'fire' has multiple meanings. If we surround it with the synonyms for the *correct* usage of the word in context, the summary will make more 'sense'
- It can also decrease our word count by grouping similar words together
- To accomplish this grouping, we are first using our preprocessing and frequency counter from unit one to get a list of the top 'n' most frequent words paired with the amount of times that they appear.
- We generate a synset for all of the words in this frequency list and increment the counter of the root word by the frequency each of the synonyms appears.
- If a synonym does not appear at all in the frequency list, then it is omitted from the final results.

### Unit 3 - A set of words constrained by POS, e.g., nouns and/or verbs

- We already have part of speech tagging as part of the wordcount. We just need to leverage that.
- Will most likely want to incorporate the important words gotten from tf-idf into this

### Unit 4 - A set of words/word stems that are discriminating features (that also are helpful in a classifier for the relevant webpages)

### Unit 5 - A set of frequent & important named entities

### Unit 6 - A set of important topics, e.g., identified using LDA

### Unit 7 - An extractive summary, as a set of important sentences, e.g., identified by clustering

### Unit 8 - A set of values for each slot matching collection semantics

### Unit 9 - A readable summary explaining the slots & values

### Unit 10 - A readable abstractive summary, e.g., from deep learning

## Useful Info

### Data

Data is split into two parts: labeled and unlabeled.

- labeled data is meant for training machine learning / deep learning modules
- unlabeled data is what we're supposed to summarize

The unlabeled files are currently stored in our home directory, in `12_Shooting_Townville_2016`. We may be asked to delete this folder later, since it's taking up unnecessary space on the cluster.

It is also stored in `/home/public/cs4984_cs5984_f18/unlabeled/data/12_Shooting_Townville_2016/`

The processed unlabeled data (the text data + timestamps and whatnot) from the WARC files is now stored in the `sentences` folder, in `JSON` format.

The labeled data is the same for all teams, stored in `/home/public/cs4984_cs5984_f18/labeled/`

### Accounts

We will have accounts on 3 different machines / clusters - Dr. Fox's Hadoop cluster (for distributed processing), and the Huckleberry and Cascades clusters (GPUs for Deep Learning).

### Easy Log-In

Typing that long username + password is a pain. There's an easier way to log in to remote systems.

- Look in `~/.ssh/`, check if you have the files `id_rsa` and `id_rsa.pub`
  - If you don't, create those using the `ssh-keygen` command
- Next, create a config file (`vim ~/.ssh/config`)
- In the config file, enter the host information for the hosts you want to ssh into. It should look like this:
    ```
    Host somehost
        HostName 1.1.1.1
        Port 22
        User [some user]

    Host dlib
        HostName hadoop.dlib.vt.edu
        Port 2222
        User [team name]
    ```
  - The `Host` can be any name you want, you just have to remember it, so make it descriptive
- Now, you can ssh into any of the hosts in the config file by typing `ssh [name of host]` (e.g. `ssh dlib`)
  - But, you still need to enter your password every time.
- Use `ssh-copy-id [name of host]` to copy your ssh key to the host's trusted keys list
- Now, you can ssh into the hsot without typing in your username, port number, or password!

### Tools

#### Hadoop

This is a tool for distributed data processing. Essentially consists of four parts:

- Distributed File-System for storing data across multiple nodes (usage: hadoop fs [shell command, eg ls, cp, etc])
- MapReduce for performing "reducing" calculations on data. E.g.: from a large list of student objects, calculate the sum of the ages. You start with a lot of data, and get a single result that 'summarizes' the data in some way.
- Hadoop Common for reading data stored in the Hadoop file system using Java. 
- YARN for managing cluster resources.

I suspect we'll be using this quite a bit, but will not be doing anything crazy with it.

Our unlabeled data is currently in the hadoop file system. It is stored in the ff directory: `/user/cs4984cs5984f18_team12/unalabed/`. This contains a WARC file and an index file into the WARC records. 

### Project Report

Our project report is on Overleaf at https://www.overleaf.com/project/5baf017bdbd72865e7330ac6

At the moment, if you need a reference added to the bibliography, you need to send me (Frank) the link to it, and hae me add it to my Mendeley account. Otherwise, we can create a group Mendeley account.
