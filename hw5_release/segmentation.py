"""
CS131 - Computer Vision: Foundations and Applications
Assignment 5
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/25/2018
Python Version: 3.5+
"""

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float
from sklearn.metrics.pairwise import euclidean_distances as ed

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.
    
    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
                vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    N, D = features.shape
    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    
    for n in range(num_iters):
        ### YOUR CODE HERE
        matrix = ed(features, centers)
        
        tmp = np.argmin(matrix, axis = 1)
        
        if np.all(tmp == assignments):
            break
        assignments = tmp
        new_centers = np.zeros_like(centers)
        for i in range(k):
            assigned_i = features[assignments == i]
            new_centers[i] = np.mean(assigned_i, axis = 0)
            #m = ed(assigned_i, assigned_i)
            #new_centers[i] = assigned_i[np.argmin(np.sum(m, axis = 0))]
        
        centers = new_centers
        ### END YOUR CODE
    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE
    return assignments
def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N) #luon dam bao [1,n_clusters]
    centers = np.copy(features) #luon dam bao co n_clusters 
    n_clusters = N
    print(centers.shape)
    matrix = ed(centers, centers)# ta se delete dan dan mang matrix
    from scipy.spatial.distance import cdist
    while n_clusters > k:
        ### YOUR CODE HERE
        
        r = np.arange(matrix.shape[0]) 
        matrix[r[:,None] >= r] = np.max(matrix) 
        center1, center2 = np.unravel_index(np.argmin(matrix), matrix.shape) #index two centers that have minimum distance
        
        if center1 > center2:
            center1, center2 = center2, center1
        assignments[assignments == center2] = center1
        assignments[assignments > center2] -= 1 #gan lai de dam bao assignments thuoc khoang [1,n_clusters]
        
        matrix = np.delete(np.delete(matrix,center2,1),center2,0) #xoa dong, cot id = center2 de thu nho matrix
        centers = np.delete(centers, center2, 0) #gan lai de dam bao
        centers[center1] = np.mean(features[assignments == center1], axis = 0)
        
        matrix[center1] = cdist(centers[center1].reshape(1,D), centers) #gan lai hang matrix
        matrix[:,center1] = matrix[center1] # gan lai cot matrix
        n_clusters -= 1
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)

    ### YOUR CODE HERE
    features = img.reshape(W * H, C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    features[:,:C] = color.reshape(H * W, C)
    features[:,C] = np.mgrid[:H,:W][0].reshape(H * W)
    features[:,-1] = np.mgrid[:H,:W][1].reshape(H * W)
    
    features = (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)
    
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    features = None
    ### YOUR CODE HERE
    color = img_as_float(img)
    features = np.zeros((H*W, C + 8))

    ### YOUR CODE HERE
    features[:,:C] = color.reshape(H * W, C)
    features[:,C] = np.mgrid[:H,:W][0].reshape(H * W)
    features[:,C + 1] = np.mgrid[:H,:W][1].reshape(H * W)
    
    features[:,C + 2] = np.gradient(color[:,:,0])[0].reshape(H * W)
    features[:,C + 3] = np.gradient(color[:,:,0])[1].reshape(H * W)
    features[:,C + 4] = np.gradient(color[:,:,1])[0].reshape(H * W)
    features[:,C + 5] = np.gradient(color[:,:,1])[1].reshape(H * W)
    features[:,C + 6] = np.gradient(color[:,:,2])[0].reshape(H * W)
    features[:,C + 7] = np.gradient(color[:,:,2])[1].reshape(H * W)
    features = (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)
    
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    #(TP + TN) / (P + N) = True / All
    tuso = np.sum(mask_gt == mask)
    mauso = mask.shape[0] * mask.shape[1]
    accuracy = tuso / mauso
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments): #NOTE: IMPORTANT FUNCTION
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
