# KMeans Clustering Algorithm 

## Introduction

### Clustering

A lot of unlabeled data is being produced nowadays that must be classified. This is where clustering comes into the picture. It is a process of finding subgroups in the data that have very similar data points within the same subgroup (cluster) and very distinct data points within different clusters. For instance, if we went to a grocery store to buy some veggies, we would see a variety of vegetables. One thing we will notice is that the veggies will be organized according to their type. Just as all the carrots will be stored in one location, so will the many varieties of potatoes be. If we look closely, we will see that each vegetable is retained within a group that corresponds to its kind, creating the clusters. This is what clustering means. 

<p align="center">
  <img width="380" height="210" src="https://559987-1802630-raikfcquaxqncofqfm.stackpathdns.com/assets/images/machine-learing/clustering/clustering01.png">
</p>

### KMeans Clustering: 
One popular clustering technique that divides the unlabeled dataset into many clusters is **```K-means```**. Each cluster, in this case, has a centroid associated with it, and *K* represents the number of pre-defined clusters. It attempts to find discrete groupings within data, where members of a group are as similar as possible to one another and as different as possible from members of other groups. K-means algorithm expects tabular data, where rows represent the observations that you want to cluster, and the columns represent attributes of the observations. The *n* attributes in each row represent a point in *n*-dimensional space. Calculating the **Euclidean distance** between these points allows one to determine how comparable the related observations are.

### KMeans Clustering has the following steps:
1. *Choosing the "k" number of clusters*
2. *Selecting some random "k" data points as centroids*
3. *Assigning the data points to the closest cluster centroid*
4. *Recomputing the centroids of newly formed clusters*

#### 1. Choosing the "k" number of clusters:
In the package, one can use the user-defined method called **```inertia()```** to know the optimal number of clusters. The process is simple, you just have to call the **inertia()** method, and it returns the optimal *k* value. There is a range of *k* values given by default inside the method, which makes the method fit the data points until it gets the optimal *k* clusters. 

#### 2. Selecting some random "k" data points as centroids:
To select the random *k* data points, there is a method known as **```centroids_of_first_iteration()```**. This method initializes the *k* number of centroids. Since it is the first iteration, a sequence of *k* numbers is randomly selected from the dataset. These chosen data points are the centroids for the first iteration.

#### 3. Assigning the data points to the closest cluster centroid:
This step can be done by calling the method **```euclidean_distance()```** which returns the distance between the centroids and the data points using the euclidean formula.
<p align="center">
  <img width="240" height="100" src="https://www.delftstack.com/img/Math/euclidean%20distance.png?ezimgfmt=rs:350x121/rscb5/ng:webp/ngcb5">

where,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x , y	=	two points in Euclidean n-space <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x<sub>i</sub> , y<sub>i</sub>	=	Euclidean vectors, starting from the origin of the space (initial point)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; n	=	n-space 
</p>

Then the **```kmeans_assignment()```** method has to be called for the allocation of each data point to its nearest centroid based on the distances and returns a list of each data point's closest centroid.

#### 4. Recomputing the centroids of newly formed clusters:
Centroid re-calculation is done using **```recalc_centroids()```** method that takes the centroids previously assigned to the data points and the data set as input. It finds the average of all the data points of each centroid and moves the centroid to that average. This method returns re-calculated/new centroids. 

## Getting Started
The package that implements the Kmeans clustering algorithm in this repository is **"KMeans_Package"**. <br>
To import the package, you can use: ```import KMeans_Package```.<br><br> 
It has modules **"\_\_init\_\_"** and **"kmeans_module"**. The package also has a directory called the **"Test"**, which consists of the unit tests. The **"kmeans_module"** module has a class named **"K_means"** that contains all the methods. <br>

There are different syntax for importing and using the package's class, some of them include, <br><br>
``` from package_name.module_name import class_name ``` <br> or<br> ``` import package_name.module_name.class_name```<br>

Now form the class if one wants implement the class's method, ``` class_name.method_name``` <br> or<br>
To import the method directly, ```from package_name.module_name.class_name import method_name```
<br><br>
For example inside the package **"KMeans_Package"**, module **"kmeans_module"**, class **"K_means"**, you want to import a method **inertia()**<br>
can be written as: ```from KMeans_Package.kmeans_module.K_means import inertia```.

## Imports
In order to use the package, the following libraries must be **installed** and imported. <br>
```
import numpy as np
import pandas as pd 
from kneed import KneeLocator
```

## Execution
1. Import the package **"KMeans_Package"** using the syntax given above.
2. Import all the required libraries as mentioned.
3. Call the **```inertia()```** method, which calculates the optimal number of clusters and returns the value.
4. Then the **```kmeans_fit()```** method should be called, which takes the value of the optimal number of clusters returned by the inertia() method as import and returns a list of all the clusters assigned to each data point as the final output.


#### Format of the Data set:
One should import the data set in **.csv** format as a DataFrame using pandas. <br>
Example: 
``` 
import pandas as pd
df = pd.DataFrame()
df=pd.read_csv("Dataset_name.csv")
```
Where, "df" is the alias of Pandas DataFrame and "pd" is alias of pandas.

#### Out of the package:
```Note:``` This part is not included in tha package. So, if needed the user has to import the required libraries and these does not come with the package.<br>

After the execution is done, one can use some visualization tools like **Matplotlib** to plot a graph that contains all the clusters assigned to each data points with their centroids.<br>
Example plot:
<p align="center">
  <img width="400" height="310" src="https://www.jcchouinard.com/wp-content/uploads/2021/10/image-54.png">
</p>

Completed!!





