# CS 4984/5984 Team 13 

- [CS 4984/5984 Team 13](#cs-49845984-team-13)
- [Workflow](#workflow)
- [What We're Doing](#what-were-doing)
  - [Unit 1](#unit-1)
- [Useful Info](#useful-info)
  - [Data](#data)
  - [Accounts](#accounts)
    - [DLRL Hadoop Cluster](#dlrl-hadoop-cluster)
    - [Huckleberry](#huckleberry)
    - [Cascades](#cascades)
  - [Easy Log-In](#easy-log-in)
  - [Tools](#tools)
    - [Hadoop](#hadoop)

# Workflow

- We need to decide how we're going to divvy up the work.
- Personally, I'm thinking we each try the units on our own (in separate git branches), and then combine what works best for our 'final' work.
  - Mostly this is because most units are 1-file scripts, which makes it hard for multiple people to collaborate on the same script.

# What We're Doing

## Unit 1

- For unit one, we will be producing a 'summary' of our data by picking the most used words in our dataset
- This will require some pre-processing:
  - The data in the JSON file (`/sentences/part-0000-....-c000.json`) is not fully cleaned. There should still be some HTML artifacts that could be gotten rid of.
  - We will want to remove stop words and punctuation.
  - We may want to use the roots of words instead of the full words
- The `nltk` module has the `word_tokenize` function and the `FreqDist` class that should do a majority of the counting work for us. It also has a database of stop words somewhere. 

# Useful Info

## Data

Data is split into two parts: labeled and unlabeled.

- labeled data is meant for training machine learning / deep learning modules
- unlabeled data is what we're supposed to summarize

The unlabeled files are currently stored in our home directory, in `12_Shooting_Townville_2016`. We may be asked to delete this folder later, since it's taking up unnecessary space on the cluster.

It is also stored in `/home/public/cs4984_cs5984_f18/unlabeled/data/12_Shooting_Townville_2016/`

The processed unlabeled data (the text data + timestamps and whatnot) from the WARC files is now stored in the `sentences` folder, in `JSON` format.

The labeled data is the same for all teams, stored in `/home/public/cs4984_cs5984_f18/labeled/`

## Accounts

We will have accounts on 3 different machines / clusters - Dr. Fox's Hadoop cluster (for distributed processing), and the Huckleberry and Cascades clusters (GPUs for Deep Learning).

Logins:

### DLRL Hadoop Cluster

- Username: cs4984cs5984f18_team12
- Password: drivers alarums
  - Note the space between the two words0
- Login: ssh cs4984cs5984f18_team12@hadoop.dlib.vt.edu -p 2222

### Huckleberry

- Need to request a login for this cluster
- 
### Cascades

- Need to request a login for this cluster

## Easy Log-In

Typing that long username + password is a pain. There's an easier way to log in to remote systems.

- Look in `~/.ssh/`, check if you have the files `id_rsa` and `id_rsa.pub`
  - If you don't, create those using the `ssh-keygen` command
- Next, create a config file (`vim ~/.ssh/config`)
- In the config file, enter the host information for the hosts you want to ssh into. It should look like this:
    ```
    Host somehost
        HostName 1.1.1.1
        Port 22
        User someuser

    Host dlib
        HostName hadoop.dlib.vt.edu
        Port 2222
        User cs4984cs5984f18_team12
    ```
  - The `Host` can be any name you want, you just have to remember it, so make it descriptive
- Now, you can ssh into any of the hosts in the config file by typing `ssh [name of host]` (e.g. `ssh dlib`)
  - But, you still need to enter your password every time.
- Use `ssh-copy-id [name of host]` to copy your ssh key to the host's trusted keys list
- Now, you can ssh into the hsot without typing in your username, port number, or password!

## Tools

### Hadoop

This is a tool for distributed data processing. Essentially consists of four parts:

- Distributed File-System for storing data across multiple nodes (usage: hadoop fs [shell command, eg ls, cp, etc])
- MapReduce for performing "reducing" calculations on data. E.g.: from a large list of student objects, calculate the sum of the ages. You start with a lot of data, and get a single result that 'summarizes' the data in some way.
- Hadoop Common for reading data stored in the Hadoop file system using Java. 
- YARN for managing cluster resources.

I suspect we'll be using this quite a bit, but will not be doing anything crazy with it.

Our unlabeled data is currently in the hadoop file system. It is stored in the ff directory: `/user/cs4984cs5984f18_team12/unalabed/`. This contains a WARC file and an index file into the WARC records. 
