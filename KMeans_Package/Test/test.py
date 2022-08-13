
import pandas as pd
import numpy as np 
from kneed import KneeLocator


#example data 1
num1 = [[1,1],[2,1],[7,1],[6,1]]
c1=[[1,1],[7,1]]
df1 = pd.DataFrame(num1, columns = ['x', 'y'])
center1= pd.DataFrame(c1, columns = ['x', 'y'])
#cluster 0 centroid(1,1) 
#data points belonging to cluster 0 [[1,1],[2,1]] 

#cluster 1 centroid (7,1)
# data points belonging to cluster 1 [[7,1],[6,1]]

#optimal clustering of the datapoints is [0,0,1,1]

#example data 2 
num2=[[1,2],[2,2],[3,9],[2,10],[8,10],[8,11],[7,9],[1,3],[2,9],[9,9],[3,10]]
c2=[[3,9],[2,2],[8,10]]
df2=pd.DataFrame(num2, columns = ['x', 'y'])
center2= pd.DataFrame(c2, columns = ['x', 'y'])

#cluster 0 centroid(3,9) 
#data points belonging to cluster 0  [3,9],[2,10],[2,9],[3,10]

#cluster 1 centroid (2,2)
# data points belonging to cluster 1 [1,2],[2,2],[1,3]

#cluster 2 centroid (8,10)
# data points belonging to cluster 2 [8,10],[8,11],[7,9],[9,9]

#optimal clustering of the datapoints is [1, 1, 0, 0, 2, 2, 2, 1, 0, 2, 0]

#example data 3
num3 = [[3,2]]
c3=[[9,10]]
df3 = pd.DataFrame(num3, columns = ['x', 'y'])
center3= pd.DataFrame(c3, columns = ['x', 'y'])

#unit testing 

#centroidsoffirstiteration returns a random k data points to be assigned as initial centroids,
#therefore this function will display a different outcome each time it is ran, so we will not perform unit test for this method.
def centroidsoffirstiteration(data,k):

    centroids=data.sample(n=k).values
    centroids = pd.DataFrame(centroids, columns = data.columns)
    return centroids


def euclidean_distance(data,centroid):
    
    c_list = []
    new_data = data.values.tolist()
    new_centroid = centroid.values.tolist()
    for i in range(len(new_centroid)):
        e_distance = []
        for j in range(len(new_data)):
            data_diff = (np.array(new_data[j]) - np.array(new_centroid[i]))**2
            distance = np.sqrt(np.sum(data_diff))
            e_distance.append(distance)
        c_list.append(e_distance)
        df2 = pd.DataFrame(np.array(c_list)).transpose()
    return df2

dist1=euclidean_distance(df1, center1)
dist2=euclidean_distance(df2, center2)

assert euclidean_distance(df1, center1).values.tolist()==[[0.0, 6.0], [1.0, 5.0], [6.0, 0.0], [5.0, 1.0]]
assert euclidean_distance(df3, center3).values.tolist()==[[10.0]] #distance between (3,2) and (9,10)

def kmeans_assignment(dist):
    
    distances=pd.DataFrame(dist).to_numpy() #converting the DatFrame to an array.
    lst=[]
    for i in distances:
        c=np.argmin(i)     #Numpy argmin is a function in python which returns the index of the minimum element from a given array along the given axis.
        lst.append(c)
    return lst


assert kmeans_assignment(dist1)==[0,0,1,1]    
assert kmeans_assignment(dist2)==[1, 1, 0, 0, 2, 2, 2, 1, 0, 2, 0]


def squarederr(a,b):
    
    error = np.sum((a-b)**2)
    return error

assert squarederr(7,2)==25
assert squarederr(6,3)==9
assert squarederr(5,1)==16

def error_calc(data,centroids):
   
    num = data.shape[0]
    centroid_errors = []
    k = centroids.shape[0]
    for ob in range(num):
        errors = np.array([])
        for centroid in range(k):
            error = squarederr(centroids.iloc[centroid, :2], data.iloc[ob,:2])
            errors = np.append(errors, error)
        centroid_error = np.amin(errors)
        centroid_errors.append(centroid_error)

    return centroid_errors

assert error_calc(df1, center1)==[0.0, 1.0, 0.0, 1.0]
assert error_calc(df2, center2)==[1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0]

#Kmeans_fit will return the same clusters and centroids each iteration but might change the numbering. 
#For example in the first iteration centroid (3,9) will be assigned as cluster 0 but in the next iteration 
#it might be assigned cluster 1. so hence it will be difficult to perform the assert function on this method
def kmeans_fit(data,k=1):
    
    # Initialize centroids and error
    centroids = centroidsoffirstiteration(data,k)
    error_list = []
    temp = True
    q = 0


    while temp:
        dist_bw_points=euclidean_distance(data,centroids)
        # Obtain centroids and error
        data['centroid'] = kmeans_assignment(dist_bw_points)
        serr=error_calc(data,centroids)
        error_list.append(sum(serr))
        # Recalculate centroids
        centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)
        data= data.drop('centroid', axis=1)

        # Check if the error has decreased
        if(len(error_list)<2):
            temp = True
        else:
            if(round(error_list[q],3) !=  round(error_list[q-1],3)):
                temp = True
            else:
                temp = False
        q = q + 1 
        
    dist_bw_points=euclidean_distance(data,centroids)
    data['centroid'] = kmeans_assignment(dist_bw_points)
    centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)
    return data['centroid'],centroids



def calculate_cost(data, centroids, cluster):
    
    sum = 0
    dataval=data.values
    centroids=centroids.values
    for i, val in enumerate(dataval):
        sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
    return round(sum,3)



cluster1=kmeans_assignment(dist1)
cluster2=kmeans_assignment(dist2)

assert calculate_cost(df1, center1, cluster1)==2.0
assert calculate_cost(df2, center2, cluster2)==9.657


def inertia(data):
   
    c=[]
    for l in range(1,10):
        clustersformed,centroidschosen =  kmeans_fit(data,l)
        cost = calculate_cost(data,centroidschosen,clustersformed)
        c.append(cost)
    kvalues = range(1,10)
    kn = KneeLocator(kvalues, c, curve='convex', direction='decreasing')
    elbow_point = kn.knee
    return elbow_point 
    
assert inertia(df2)==3








