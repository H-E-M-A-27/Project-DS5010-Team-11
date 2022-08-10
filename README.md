# KMeans Clustering Algorithm 

## Introduction

### Clustering

A lot of unlabeled data is being produced nowadays that must be classified. This is where clustering comes into the picture. It is a process of finding subgroups in the data that have very similar data points within the same subgroup (cluster) and very distinct data points within different clusters. For instance, if we went to a grocery store to buy some veggies, we would see a variety of vegetables. One thing we will notice is that the veggies will be organized according to their type. Just as all the carrots will be stored in one location, so will the many varieties of potatoes be. If we look closely, we will see that each vegetable is retained within a group that corresponds to its kind, creating the clusters. This is what clustering means. 

<p align="center">
  <img width="380" height="210" src="https://559987-1802630-raikfcquaxqncofqfm.stackpathdns.com/assets/images/machine-learing/clustering/clustering01.png">
</p>

### KMeans Clustering: 
One popular clustering technique that divides the unlabeled dataset into many clusters is **```K-means```**. Each cluster in this case has a centroid associated with it, and *K* represents the number of pre-defined clusters. It attempts to find discrete groupings within data, where members of a group are as similar as possible to one another and as different as possible from members of other groups. K-means algorithm expects tabular data, where rows represent the observations that you want to cluster, and the columns represent attributes of the observations. The *n* attributes in each row represent a point in *n*-dimensional space. Calculating the **Euclidean distance** between these points allows one to determine how comparable the related observations are.

### KMeans Clustering has the following steps:
1. *Choosing the "k" number of clusters*
2. *Selecting some random "k" data points as centroids*
3. *Assigning the data points to the closest cluster centroid*
4. *Recomputing the centroids of newly formed clusters*

#### 1. Choosing the "k" number of cluster:
In the package one can use the user-defined method called **```inertia()```** to know the optimal number of clusters. The process is simple, you just have to call the **inertia()** method and it returns the optimal *k* value. There is a range of *k* values given by default inside the method which makes the method to fit the data points until it gets the optimal *k* clusters. 

#### 2. Selecting some random "k" data points as centroids:
To select the random *k* data points there is a method known as **```centroids_of_first_iteration()```**. This method initializes *k* number of centroids. Since it is the first iteration, a sequence of *k* numbers are randomly selected from the dataset. These chosen datapoints are the centroids for the first iteration.

#### 3. Assigning the data points to the closest cluster centroid:
This step can be don by calling the method **```euclidean_distance()```** which returns the distance between the centroids and the data points using the euclidean formula.
<p align="center">
  <img width="240" height="100" src="https://www.delftstack.com/img/Math/euclidean%20distance.png?ezimgfmt=rs:350x121/rscb5/ng:webp/ngcb5">

where,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x , y	=	two points in Euclidean n-space <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x<sub>i</sub> , y<sub>i</sub>	=	Euclidean vectors, starting from the origin of the space (initial point)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; n	=	n-space 
</p>

Then the **```kmeans_assignment()```** method has to be called for allocation of each data point to its nearest centroid based on the distances, and returns a list of each data point's closest centroid.

#### 4. Recomputing the centroids of newly formed clusters:
Centroid re-calculation is done using **```recalc_centroids()```** method that takes the centroids previously assigned to the data points and the data set as input. It finds the average of all the data points of each centroid and moving the centroid to that average. This method returns re-calculated/new centroids. 


## Getting Started
The package that implements Kmeans clustering algorithm in this repository is **"KMeans_Package"**. <br>
It has modules **"\_\_init\_\_"** and **"kmeans_module"**. The package also has a directory called the **"Test"** which consists of the unit tests. The **"kmeans_module"** module has a class named **"K_means"** that contains all the methods. <br>

There are different syntax for importing and using the package's class, some of them include,<br><br>
``` from package_name.module_name import class_name ``` <br>or<br> ``` import package_name.module_name.class_name```<br>

Now form the class if one wants implement the class's method, ``` class_name.method_name``` <br>, or<br>
To import the method directly, ```from package_name.module_name.class_name import method_name```
<br><br>
For example inside the package **"KMeans_Package"**, module **"kmeans_module"**, class **"K_means"**, you want to import a method **inertia()**<br>
can be written as: ```from KMeans_Package.kmeans_module.K_means import inertia```.


## Imports
In order to use the package the following libraries must be imported. <br>
```
import numpy as np
import pandas as pd 
from kneed import KneeLocator
```









