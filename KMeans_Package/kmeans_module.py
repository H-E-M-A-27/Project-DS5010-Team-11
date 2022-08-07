import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from kneed import KneeLocator

class K_means:

  def __init__(self):
      pass
      
  def initialize_centroids(k, data):
      centroids=data.sample(n=k).values
      centroids = pd.DataFrame(centroids, columns = data.columns)
      return centroids


  def euclidean_distance(centroid, data):
      cent_list = []
      new_data = data.values.tolist()
      new_centroid = centroid.values.tolist()
      for i in range(len(new_centroid)):
          e_distance = []
          for j in range(len(new_data)):
              data_diff = (np.array(new_data[j]) - np.array(new_centroid[i]))**2
              distance = np.sqrt(np.sum(data_diff))
              e_distance.append(distance)
          cent_list.append(e_distance)
          df2 = pd.DataFrame(np.array(cent_list)).transpose()
      return df2


  def kmeans_assignment(centroids, points, dist):
        distances=pd.DataFrame(dist).to_numpy()
        lst=[]
        for i in distances:
            c=np.argmin(i)
            lst.append(c)
        return lst


  def recalc_centroids(clusters, data):
        new_centroids = []
        new_df = pd.concat([pd.DataFrame(data), pd.DataFrame(clusters, columns=['cluster'])],axis=1)
        for c in set(new_df['cluster']):
            curr_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
            cluster_mean = curr_cluster.mean(axis=0)
            new_centroids.append(cluster_mean)     
        return new_centroids


  def calculate_error(a,b):
        error = np.square(np.sum((a-b)**2))
        return error


  def error_calc(data, centroids):
        n_observations = data.shape[0]
        centroid_errors = []
        k = centroids.shape[0]

        for observation in range(n_observations):

            # Calculate the errror
            errors = np.array([])
            for centroid in range(k):
                error = K_means.calculate_error(centroids.iloc[centroid, :2], data.iloc[observation,:2])
                errors = np.append(errors, error)

            centroid_error = np.amin(errors)

            centroid_errors.append(centroid_error)

        return (centroid_errors)
      
      
      
  def kmeans(data, k):
        # Initialize centroids and error
        centroids = K_means.initialize_centroids(k, data)
        error = []
        compr = True
        i = 0
        dist=K_means.euclidean_distance(centroids, data)

        while(compr):
            # Obtain centroids and error
            data['centroid'] = K_means.kmeans_assignment(centroids,data,dist)
            iter_error=K_means.error_calc(data, centroids)
            error.append(sum(iter_error))
            # Recalculate centroids
            newcentroids = K_means.recalc_centroids(data['centroid'],data)

            # Check if the error has decreased
            if(len(error)<2):
                compr = True
            else:
                if(round(error[i],3) !=  round(error[i-1],3)):
                    compr = True
                else:
                    compr = False
            i = i + 1 

        data['centroid'] = K_means.kmeans_assignment(centroids,data,dist)
        iter_error=K_means.error_calc(data, centroids)
        newcentroids = K_means.recalc_centroids(data['centroid'],data)
        return (data['centroid'], iter_error, newcentroids,centroids)


  def calculate_cost(data, centroids, cluster):
        sum = 0
        X=data.values
        centroids=centroids.values
        for i, val in enumerate(X):
          sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
        return sum


  def inertia(data):
        c=[]
        for l in range(1,10):
            clus, errors, centroids,j =  K_means.kmeans(data,l)
            cost = K_means.calculate_cost(data, j,clus)
            c.append(cost)
        k_range = range(1,10)
        kn = KneeLocator(k_range, c, curve='convex', direction='decreasing')
        elbow_point = kn.knee
        return elbow_point
