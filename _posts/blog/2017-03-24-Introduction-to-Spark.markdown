---
layout: post
title:  "Introduction to Spark"
date:   2017-03-24 
categories: blog
---


I have been playing around with spark recently to become more familiar with big data and in particular the Hadoop framework. This post will be an introductory tutorial into using spark through the cloudera VM.

You will need a couple of things before you get started. First you will need to install Oracle virtualbox and this can be downloaded from [here](https://www.virtualbox.org/wiki/Downloads). The next thing we need is the [Cloudera quickstart VM](http://www.cloudera.com/downloads/quickstart_vms/5-8.html)
we will be running spark.

Since I am currently taking one of the big data courses on Coursera, Hadoop platform and application framework, I will implement one of the assignments from the course. A simple join between two datasets using spark via pyspark. Upon starting up the VM machine you will see something similar to the following (I am using a linux distribution btw):


![cloudera](/assets/img/cloudera-300x227.png)

&nbsp;

Ok, so the first thing we need is a dataset. We will create a toy dataset just for testing purposes. In a later tutorial I will use one of that bigger data sets that come pre-installed with cloudera.

```python
gedit join_FileA.txt
```

We will insert the following data to join_FileA.txt:

able,991
about,11
burger,15
actor,22

Similarly for our second file we can the following

```python
gedit join_FileB.txt
```

and input the following data:

Jan-01 able,5
Feb-02 about,3
Mar-03 about,8
Apr-04 able,13
Feb-22 actor,3
Feb-23 burger,5
Mar-08 burger,2
Dec-15 able,100

We are next going to save these files into a folder I have created called join in the HDFS.
This is done using the appropriate terminal commands. Below we first view the contents
of the folder using the -ls command which lists the files and then we move the .txt files
into the HDFS using the put command and assuming our data is in the cloudera folder. 
```python
hdfs dfs -ls /user/cloudera/input

hdfs dfs -put /home/cloudera/join_FileA.txt  /user/cloudera/input/join/

hdfs dfs -put /home/cloudera/join_FileB.txt  /user/cloudera/input/join/

```


After we have saved these files in appropriate locations we proceed to install ipython on our virtual machine.
In a terminal type the following

```python
sudo easy_install ipython==1.2.1
```

After the install we can then launch a pyspark shell very easily using the command:

```python
PYSPARK_DRIVER_PYTHON=ipython pyspark
```

You should see the following in the terminal console:

<img src="http://176.32.230.1/ecomlblog.com/wordpress/wp-content/uploads/2017/01/Pyspark-300x229.png" alt="pyspark" width="500" height="350" class="alignnone size-medium wp-image-13" />

Our goal is to perform a SQL like join  where our data gets aggregate together with the word then the amount of times that word has appeared on a particular date. In the spark console we can do a few commands to load in the data as well as look at the formatting. 

```python

fileA = sc.textFile("input/join/join_FileA.txt")

fileA.collect()

fileB = sc.textFile("input/join/join_FileB.txt")
fileB.collect()
```

The output should for both commands should be this:
Out[]: [u'able,991', u'about,11', u'burger,15', u'actor,22']

Out[]: 
[u'Jan-01 able,5',
 u'Feb-02 about,3',
 u'Mar-03 about,8 ',
 u'Apr-04 able,13',
 u'Feb-22 actor,3',
 u'Feb-23 burger,5',
 u'Mar-08 burger,2',
 u'Dec-15 able,100']


Since we are using Hadoop we need to split our operations into a Mapper and a Reducer. For a quick revision on how MapReduce actually works see the following [page](https://www.tutorialspoint.com/hadoop/hadoop_mapreduce.htm)

Since in the real world we are likely dealing with vast amounts of data that is distributed across different servers we need a way to group the data together. This is what the Mapper will do. It groups data into key value pairs which is then passed to the Reducer which combines this data into an output.

Ok, so now what we know a bit about MapReduce, let's define our mapper function. Let's call it split_fileA. We need to have our function split the input line into a word and a count separated by a comma and return that.

```python
def split_fileA(line):
  line = line.strip()
  key_value = line.split(",")
  word = key_value[0]
  count = int(key_value[1])
  return(word,count)
```

We can test this to see if it works

```python
test_line ="could, 10001"
split_fileA(test_line)
```

this returns Out[]: ('could', 1001)
which is exactly what we want.

Now we can call the spark map method on using our defined mapper on 
fileA. We then use the collect function to display the results.


```python
fileA_data = fileA.map(split_fileA)
fileA_data.collect()
```

This gives us a new file with the following format.

Out[]: [(u'able', 991), (u'about', 11), (u'burger', 15), (u'actor', 22)]

Ok, that's one mapper function down and one to go. The mapper for fileB is slightly different as we have more values to deal with but it should be simple enough to implement.


```python
def split_FileB(line):
   date_word_count = line.split(",")
   count = date_word_count[1]
   date_word = date_word_count[0].split("")
   date = date_word[0]
   word = date_word[1]
   return(word, date +"" + count)

fileB_data = fileB.map(split_FileB)
fileB_data.collect()
```

Our output should be the following:

Out[]: 
[(u'able', u'Jan-01 5'),
 (u'about', u'Feb-02 3'),
 (u'about', u'Mar-03 8 '),
 (u'able', u'Apr-04 13'),
 (u'actor', u'Feb-22 3'),
 (u'burger', u'Feb-23 5'),
 (u'burger', u'Mar-08 2'),
 (u'able', u'Dec-15 100')]

The final step is to join the datasets using the spark join command.
The join command works by joining a dataset of (k,v) pairs with another dataset of (k,w) pairs and returns a dataset with (k,(v,w)) pair.


```python
joined_file = fileB_data.join(fileA_data)
joined_file.collect()
```

our output returns:

[(u'able', (u'Jan-01 5', 991)),
 (u'able', (u'Apr-04 13', 991)),
 (u'able', (u'Dec-15 100', 991)),
 (u'burger', (u'Feb-23 5', 15)),
 (u'burger', (u'Mar-08 2', 15)),
 (u'about', (u'Feb-02 3', 11)),
 (u'about', (u'Mar-03 8', 11)),
 (u'actor', (u'Feb-22 3', 22))]

This concludes this simple introductory tutorial into using spark. In a later tutorial I will try my hand at a larger dataset and will implement some popular machine learning algorithms to analyse the data.

