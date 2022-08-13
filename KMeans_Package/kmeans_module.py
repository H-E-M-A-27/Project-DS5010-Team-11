import numpy as np
import pandas as pd
from kneed import KneeLocator

class K_means:

  def __init__(self,data):
      self.data=data
      
  def centroidsoffirstiteration(self,k):
      '''
      This method initializes k number of centroids. Since it is the first iteration, a sequence of
      k numbers are randomly selected from the dataset. These chosen datapoints are the centroids for the first
      iteration.
      
      Parameters:
      ----------
      k (integer): the number of clusters 

      Returns:
      -------
      centroids (Type: dataframe): a dataframe of k centroids which are initialized

      '''
      centroids=self.data.sample(n=k).values
      centroids = pd.DataFrame(centroids, columns = self.data.columns)
      return centroids


  def euclidean_distance(self,centroid):
      '''
      Euclidean distance is the length of the line segment or the distance between two points. 
      We find the euclidean distance from each point to the k centroids. 
      This method returns the distance between the centroids and the data using the euclidean formula as a dataframe.
      Parameters:
      ----------
      centroid : It has the k centroids as a dataframe.
      
      Returns:
      -------
      df2 (Type: dataframe): The distance between each point to the centroid.
      '''
      
      c_list = []
      new_data = self.data.values.tolist()
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


  def kmeans_assignment(self,dist):
      '''
      Method for allocation of each data point to its nearest centroid based on the distances.
      Once the distances from cetroids to the data points are calculated using the method "euclidean_distance",
      the data point is assigned to the closest centroid. 

      Parameters:
      ----------
      dist: It is a pandas dataframe which is later converted into a array(numpy.ndarray). 
            This parameter consistes of distance between all datapoints to all the centroids.
      
      Returns:
      -------
      lst (Type: list): A list that consists of each data point's closest centroid.  
      '''
      distances=pd.DataFrame(dist).to_numpy() #converting the DatFrame to an array.
      lst=[]
      for i in distances:
          c=np.argmin(i)     #Numpy argmin is a function in python which returns the index of the minimum element from a given array along the given axis.
          lst.append(c)
      return lst


  def recalc_centroids(self,prev_centroids):
      '''
      recalc_centroids() method takes the centroids previously assigned to the data points and the data set as input.
      It finds the average of all the data points of each centroid and moving the centroid to that average. This method 
      returns re-calculated/new centroids. 

      Parameters:
      ----------
      prev_centroids (Type: integers): A list that contains the previously assigned centroids to each data point.

      Returns:
      -------
      new_centroids (Type: numbers): A list of newly calculated centroids based on the mean.
      '''
      new_centroids_list = []
      #concatinating the DataFrame of data and previous centroids.
      new_df = pd.concat([pd.DataFrame(self.data), pd.DataFrame(prev_centroids, columns=['cluster'])],axis=1)
      for c in set(new_df['cluster']):
          point_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
          meanofcluster = point_cluster.mean(axis=0) #caculating the mean.
          new_centroids_list.append(meanofcluster)
          nc=pd.DataFrame(new_centroids_list, columns = self.data.columns)
      return nc


  def squarederr(self,a,b):
      '''
      squarederr() calculates the error between two data points. Error id defined as the sum of the
      squares of the difference between two points. In other words, the square of the euclidean distance is
      known as error. Error is needed in recalculating the new centroids in every iteration
      
      Parameters:
      ----------
      a and b (Type: integers): two integer values between which the error needs to be calculated

      Returns:
      -------
      error (Type: numbers): a number calculated using the error formula

      '''
      error = np.sum((a-b)**2)
      return error


  def error_calc(self,centroids):
      '''
      error_calc() calculates the error between the list of data points and the assigned centroids for 
      that iteration. 
      
      Parameters:
      ----------
      centroids (Type: dataframe): a dataframe of k centroids which are initialized

      Returns:
      -------
      centroid_errors (Type: list): it returns a list of errors calculated between each data point and centroid

      '''
      
      num = self.data.shape[0]
      centroid_errors = []
      k = centroids.shape[0]
      for ob in range(num):
          errors = np.array([])
          for centroid in range(k):
              error = self.squarederr(centroids.iloc[centroid, :2], self.data.iloc[ob,:2])
              errors = np.append(errors, error)
          centroid_error = np.amin(errors)
          centroid_errors.append(centroid_error)

      return centroid_errors
      
      
  def inertia(self):
      '''
      inertia() calculates the optimal number of clusters. It takes into consideration the cost with
      different k values and finds the most optimal one. To locate the optimal k value we have used the
      elbow method. It picks the elbow of the curve used to plot the k values and their error cost.
      
      Parameters:
      ----------
      None

      Returns:
      -------
      elbow_point (Type: number): it returns a optimal number for k 

      '''
      c=[]
      for l in range(1,10):
          clustersformed,centroidschosen =  self.kmeans_fit(l)
          cost = self.calculate_cost(centroidschosen,clustersformed)
          c.append(cost)
      kvalues = range(1,10)
      kn = KneeLocator(kvalues, c, curve='convex', direction='decreasing')
      elbow_point = kn.knee
      return elbow_point    
  

  def calculate_cost(self, centroids, cluster):
      '''
      This method calculates the error after clustering. It calculates the sum of the errors between all the 
      centroids and the data points assigned to each cluster. This sum of all error cost of different clusters
      gives the total error of that particular iteration. since we intend to decrease the intra cluster distance
      and maximize the inter cluster distance, the total error needs to be as low as possible.
      
      Parameters:
      ----------
      centroids (Type: dataframe): a dataframe of k centroids which are initialized
      
      cluster(series/list): a list of all the clusters assigned to each data point


      Returns:
      -------
      sum (Type: number): it returns sum of the cost of each iteration

      '''
      sum = 0
      dataval=self.data.values
      centroids=centroids.values
      for i, val in enumerate(dataval):
          sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
      return sum
    
  def kmeans_fit(self,k=1):
      '''
      This method collectively calls the rest of the methods to perform k means algorithm.
      After the first iteration when the data points are assigned to a cluster, the total error is calculated 
      for each iteration. The while loop runs as long as the total error is still decreasing and the centroids
      are recalculated. This process is repeated until the error no longer decreases.
      Parameters:
      ----------
      k (integer): if the user enters a value for k, then k is assigned that value or else the default k value
      is 1.


      Returns:
      -------
      data['centroid'] (Type: series/list): a list of all the clusters assigned to each data point
      centroids (Type: dataframe): a dataframe of k centroids which are calculated for the last iteration

      '''
      # Initialize centroids and error
      centroids = self.centroidsoffirstiteration(k)
      error_list = []
      temp = True
      q = 0


      while temp:
          dist_bw_points=self.euclidean_distance(centroids)
          # Obtain centroids and error
          self.data['centroid'] = self.kmeans_assignment(dist_bw_points)
          serr=self.error_calc(centroids)
          error_list.append(sum(serr))
          # Recalculate centroids
          centroids = self.data.groupby('centroid').agg('mean').reset_index(drop = True)
          self.data= self.data.drop('centroid', axis=1)

          # Check if the error has decreased
          if(len(error_list)<2):
              temp = True
          else:
              if(round(error_list[q],3) !=  round(error_list[q-1],3)):
                  temp = True
              else:
                  temp = False
          q = q + 1 
          
      dist_bw_points=self.euclidean_distance(centroids)
      self.data['centroid'] = self.kmeans_assignment(dist_bw_points)
      centroids = self.data.groupby('centroid').agg('mean').reset_index(drop = True)
      return self.data['centroid'],centroids

