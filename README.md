# K-means
K-Meas Clustering Scratch Coding by Python

Task: pick one of the digits (e.g. the "ones"), which gives you a dataset of 200 image vectors. Carry out a K-means 
clustering on your chosen sample, setting K = 1 (!), 2, 3, and 200 in four runs of this algorithm. Generate visualizations of the
images that are coded in the respective codebook vectors that you get (for the K = 200 case, only visualize a few). Discuss what you see.

K means Clustering Interpretation:

K = 1;
The mathematical interpretation is to take the average of 200 hundred selected image vectors. Since we are taking the average of these 200 hundred similar vectors, at the end we are expecting to obtain a similar image to these 200 vectors.

K = 2;

The mathematical interpretation is to take two centroids which minimize the euclidean distance of two groups of digits 1(one) around them. We obtained two vectors each of them is consisting of the average of two groups of selected image vectors.

K = 3;

The mathematical interpretation is to take 3 centroids which minimize the euclidean distance of 3 groups of 200 digits 1(one) around them.  As a result we expected to obtain three vectors that are the average of each three group respectively.

K = 200;

The mathematical interpretation is to take 200 centroids which minimize the euclidean distance of 200 groups of digits 1(one) around them. Actually that is the situation in which every digit is itself a centroid. So we obtain the digits themselves. 
