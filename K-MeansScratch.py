# -*- coding: utf-8 -*-

"Machine Learning -K Means from Scratch-Diego&Yasin"

import os
print(os.getcwd())
os.chdir(r'....\DigitsBasicRoutines\DigitsBasicRoutines')
print(os.getcwd())
print(os.listdir())

import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

#Loading the dataset
X=np.loadtxt('mfeat-pix.txt')
X

#Plot the ten first digits

def plot_database(database):
    for i in range (10):
        for j in range (10):
            plt.subplot(1, 10, j+1) #make a 1 by 10 grid, and paste in each grid the jth element, note that j starts with zero, therefore jth+1 for the first element
            databases=database[j + 200 * i,].reshape(16, 15) #select the first row,240 digits, of the digits array, and reshape them in a matrix 16x15 
            plt.imshow(databases, cmap='gray') #plot a grid, where the number is the "gray" value
            plt.axis('off') #Turn plot axis off
        plt.show() # show the graph
plot_database(database=X)

#X is data represented as numpy array

def k_means (X, K):
    nrow = X.shape[0] # the shape function with [0] returns the numbers of rows
    ncol = X.shape[1] # the shape function with [1] returns the number of columns
    
    #pick K random data points as initial centroids
    initial_centroids = np.random.choice(nrow, K, replace=False) #the np.random.choice function is for choosing from nrows, K random elements without replacement
    
    centroids = X[initial_centroids] #selecting the inicial centroids from the numpu array
    centroids_old=np.zeros((K, ncol)) #Creation of an array filled with zeros with k columns and ncol rows
    
    cluster_assignments=np.zeros(nrow) #Creation of an array filled with zeros with nrow columns
    while (centroids_old != centroids).any(): #When centroids old are unequal to centroids new continue, if they are equal stop
        centroids_old=centroids.copy() # Change the previous centroids with the new centrois
        
        #compute de distances between data points and centroids
        dist_matrix = distance_matrix(X, centroids, p=2) #Euclidean distance between the data points and centroids
        
        for i in np.arange(nrow):
            #find closest centroid
            d = dist_matrix[i] #matrix of distances
            closest_centroid=(np.where(d==np.min(d)))[0][0] #Extracting the closest centroid to each point
            
            #associate data points with closest centroid
            cluster_assignments[i] = closest_centroid #Assign the closest centroid
            
        #recompute centroids
        for k in np.arange(K):
            Xk = X[cluster_assignments == k] #
            centroids [k] = np.apply_along_axis(np.mean, axis=0, arr=Xk)
    
    return (centroids, cluster_assignments)

#-------------------------------------------------------------------------------
X=X[200:400,0:240]
X
X.shape[0]
X.shape[1]

k_means (X,1)

centroid1, cluster = k_means(X,1)
plt.imshow(centroid1.reshape(16, 15), cmap='gray')


centroid2, cluster = k_means(X,2)

for i in range(centroid2.shape[0]):
    centroid_i = centroid2[i]
    plt.imshow(centroid_i.reshape(16, 15), cmap='gray')

plt.imshow(centroid2[0].reshape(16, 15), cmap='gray')
plt.imshow(centroid2[1].reshape(16, 15), cmap='gray')

centroid3, cluster = k_means(X,3)

for i in range(centroid3.shape[0]):
    centroid_i = centroid3[i]
    plt.imshow(centroid_i.reshape(16, 15), cmap='gray')

plt.imshow(centroid3[0].reshape(16, 15), cmap='gray')
plt.imshow(centroid3[1].reshape(16, 15), cmap='gray')
plt.imshow(centroid3[2].reshape(16, 15), cmap='gray')

centroid200, cluster = k_means(X,200)

plt.imshow(centroid200[0].reshape(16, 15), cmap='gray')
plt.imshow(centroid200[1].reshape(16, 15), cmap='gray')
plt.imshow(centroid200[2].reshape(16, 15), cmap='gray')
plt.imshow(centroid200[15].reshape(16, 15), cmap='gray')
plt.imshow(centroid200[25].reshape(16, 15), cmap='gray')
plt.imshow(centroid200[35].reshape(16, 15), cmap='gray')
plt.imshow(centroid200[45].reshape(16, 15), cmap='gray')
