from pylab import *
from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random

#Data is saved in a .csv file
df_full = pd.read_csv("ionosphere.csv",header=None)

df_full.rename(columns={34: 'label'}, inplace=True)
df_full['label'] = df_full.label.astype('category')
encoding = {'g': 1, 'b': 0}
df_full.label.replace(encoding, inplace=True)


columns = list(df_full.columns)
features = columns[:len(columns) - 1]
# class_labels = list(df_full[columns[-1]])
df = df_full[features]
#Number of attributes except class label
num_attr = len(df.columns) - 1
#Number of clusters
k = 2
# Maximum number of iterations
MAX_ITER = 100
# Number of samples
n = len(df)  # the number of row
# fuzzy parameter
m =2


# Initialize the fuzzy matrix U
def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]  #First normalization
        membership_mat.append(temp_list)
    return membership_mat


# 
def calculateClusterCenter(membership_mat):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(k):
        x = cluster_mem_val_list[j]
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]  # Each dimension must be calculated.
        cluster_centers.append(center)
    return cluster_centers


#Update membership
def updateMembershipValue(membership_mat, cluster_centers):
    #    p = float(2/(m-1))
    data = []
    for i in range(n):
        x = list(df.iloc[i])  # Take out each line of data in the file
        data.append(x)
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j] / distances[c]), 2) for c in range(k)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat, data


# Get cluster results
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # 
    membership_mat = initializeMembershipMatrix()
    curr = 0
    cent_temp = [[0,1,0.8759,0.04859,0.85243,0.02306,0.83398,-0.37708,1,0.0376,0.85243,0.17755,-0.59755,0.44945,0.60536,0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,0.46168,0.21785,0.3409,0.42267,-0.54487,0.18641,-0.453],[0,1,0.01864,-0.08459,0,0,0,0.125,0.1234,-0.2681,-0.5674,-0.48752,0,0,-0.33656,0.38602,-0.37133,0.15018,-0.63728,-0.22115,0,0,0,0,-0.14803,-0.01326,0.20645,-0.02294,0,0,0.16595,0.24086,-0.08208,0.38065]]
    while curr <= MAX_ITER:  # The maximum number of iterations
         if(curr == 0):
            cluster_centers = cent_temp
            print("Intial Randomly selected Cluster Centers:")
            print(np.array(cluster_centers))
         else:
            cluster_centers = calculateClusterCenter(membership_mat)
         membership_mat, data = updateMembershipValue(membership_mat, cluster_centers)
         cluster_labels = getClusters(membership_mat)
         curr += 1
    #print(membership_mat)
    return cluster_labels, cluster_centers, data, membership_mat

labels, centers, data, membership = fuzzyCMeansClustering()
print('\nLabels:')
print(labels)
print('\nFinal Centers:')
print(centers)
import itertools 

def jaccard(labels1, labels2):
    """
    Computes the Jaccard similarity between two sets of clustering labels.
    The value returned is between 0 and 1, inclusively. A value of 1 indicates
    perfect agreement between two clustering algorithms, whereas a value of 0
    indicates no agreement.
    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)
    @param labels1 iterable of cluster labels
    @param labels2 iterable of cluster labels
    @return the Jaccard similarity value
    """
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)
jaccard(label,df_full['label'])
