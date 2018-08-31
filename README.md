# CS 4984/5984 Team 13 

# Useful Info

## Data

Data is split into two parts: labeled and unlabeled.

- labeled data is meant for training machine learning / deep learning modules
- unlabeled data is what we're supposed to summarize

The unlabeled files are currently stored in our home directory, in `12_Shooting_Townville_2016`. We may be asked to delete this folder later, since it's taking up unnecessary space on the cluster.

It is also stored in `/home/public/cs4984_cs5984_f18/unlabeled/data/12_Shooting_Townville_2016/`

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

## Tools

### Hadoop

This is a tool for distributed data processing. Essentially consists of four parts:

- Distributed File-System for storing data across multiple nodes (usage: hadoop fs [shell command, eg ls, cp, etc])
- MapReduce for performing "reducing" calculations on data. E.g.: from a large list of student objects, calculate the sum of the ages. You start with a lot of data, and get a single result that 'summarizes' the data in some way.
- Hadoop Common for reading data stored in the Hadoop file system using Java. 
- YARN for managing cluster resources.

I suspect we'll be using this quite a bit, but will not be doing anything crazy with it.

Our unlabeled data is currently in the hadoop file system. It is stored in the ff directory: `/user/cs4984cs5984f18_team12/unalabed/`. This contains a WARC file and an index file into the WARC records. 
