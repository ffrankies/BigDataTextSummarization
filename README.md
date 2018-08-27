# CS 4984/5984 Team 13 

# Useful Info

## Data

Data is split into two parts: labeled and unlabeled.

- labeled data is meant for training machine learning / deep learning modules
- unlabeled data is what we're supposed to summarize

## Accounts

We will have accounts on 3 different machines / clusters - Dr. Fox's Hadoop cluster (for distributed processing), and the Huckleberry and Cascades clusters (GPUs for Deep Learning).

Logins:

### DLRL Hadoop Cluster

- Username: cs4984cs5984f18_event13
- Password: drivers alarums
  - Note the space between the two words
- Login: ssh cs4984cs5984f18_event13@hadoop.dlib.vt.edu -p 2222

### Huckleberry

- Need to request a login for this cluster

### Cascades

- Need to request a login for this cluster

## Tools

### Hadoop

This is a tool for distributed data processing. Essentially consists of four parts:

- Distributed File-System for storing data across multiple nodes (usage: hadoop fs [shell command, eg ls, cp, etc])
- MapReduce for performing "reducing" calculations on data. E.g.: from a large list of student objects, calculate the sum of the ages. You start with a lot of data, and get a single result that 'summarizes' the data in some way.
- Hadoop Common for reading data stored in the Hadoop file system using Java. 
- YARN for managing cluster resources.

I suspect we'll be using this quite a bit, but will not be doing anything crazy with it.
