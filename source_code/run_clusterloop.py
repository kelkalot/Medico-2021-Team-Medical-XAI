
#Importing libraries
import random, shutil, os
import copy
from numpy.core.numeric import tensordot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from torch.utils.data.dataset import Dataset
#from extract_images import extract_images
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets, models

import sys


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
#Selected 100 images for each label class (Kvasir)
#Selected ALL unlabelled images not in labelled image dataset (Hyperkvasir)
#Extracted features from LIRE unlabelled images
#Extracted features from LIRE labelled images


#Test combination of EdgeHistogram, Tamura, LuminanceLayout and SimpleColorHistogram
#Location of csv-files with extracted features:
polyps_file = '../global-features/features_50ClustersModel/polyp_tiles_globalfeatures.csv'
nonpolyps_file = '../global-features/features_50ClustersModel/nonpolyp_tiles_globalfeatures.csv'
unlabeled_file = '../global-features/features_50ClustersModel/unlabeled_tiles_globalfeatures.csv'


### Preprocessing to create the large df:
#Find correct number of columns:
test_df = pd.read_csv(polyps_file, skiprows=1)
n_columns = test_df.shape[1]
column_names = []
for c in range(1,n_columns):
    column_names.append('Feature_' + str(c))
column_names.append('Filepath')

#Read the csv-files to df:
large_df = pd.read_csv(polyps_file, skiprows=1, names = column_names)
large_df['Label'] = 'polyps'
nonpolyps_df = pd.read_csv(nonpolyps_file, skiprows=1,names = column_names)
nonpolyps_df['Label'] = 'nonpolyps'
large_df = pd.concat([large_df, nonpolyps_df], ignore_index=True, axis = 0)
unlabeled_df = pd.read_csv(unlabeled_file, skiprows = 1, names = column_names)
unlabeled_df['Label'] = 'unlabeled'
#Extract only features from the unlabeled images:
unlabeled_df = unlabeled_df.iloc[np.where(unlabeled_df['Label']=='unlabeled')[0].tolist(),:]

#Checks that the images not already exist in large_df:
remove_list = np.where(unlabeled_df['Filepath'].isin(large_df['Filepath']))[0].tolist()
unlabeled_df.drop(remove_list, axis = 0, inplace = True)
#unlabeled_df['label'] = 'unlabeled'
large_df = pd.concat([large_df, unlabeled_df],ignore_index = True, axis = 0)

print('Large DF:')
print(large_df.iloc[:, -3:])
#Extract image name from filepath:

large_df['image'] = 'lol'
for i in range(large_df.shape[0]):
    image_text = str(large_df['Filepath'][i])
    #Splits on /
    pieces = image_text.split('/')
    #The last part equals the image:
    image_name = pieces[-1]
    large_df.iloc[i,-1] = image_name

print(large_df.iloc[:,-3:])
print('Number of unlabelled images:', large_df.iloc[np.where(large_df['Label']=='unlabeled')[0].tolist(),:].shape[0])

#Write to csv:
#large_df.to_csv('polyp_df_clustering.csv', index = False)


'''
#Import the data file:
df = pd.read_csv('polyp_df_clustering.csv')
print(df.iloc[:,-3:])
print('Number of unlabelled images:', df.iloc[np.where(df['Label']=='unlabeled')[0].tolist(),:].shape[0])
print(df.shape)
#Remove rows that contain images not saved on my computer
our_image_path = '/Users/andreastoras/Desktop/Kvasir_datasets/hyperkvasir_unlabeled/images'
our_images = os.listdir(our_image_path)
#Checks that the images are not already in large_df:
keep_list = np.where(df['image'].isin(our_images))[0].tolist()
keep_df = df.iloc[keep_list, :]
large_df = pd.concat([df.iloc[:200,:], keep_df],ignore_index = True, axis = 0)
print(large_df.shape)
print('Number of labelled images:', large_df.iloc[np.where(df['Label']!='unlabeled')[0].tolist(),:].shape[0])
'''



def run_clusterloop(large_df, num_clusters = 5):
    #Create clusters
    my_cluster = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0, max_iter=300, algorithm='auto')
    
    #Fit clusters on all images!
    my_cluster.fit_transform(large_df.iloc[:,:-3])

    #Predict clusters for all images:
    pred_clusters = my_cluster.predict(large_df.iloc[:,:-3])

    #Create column equal to predicted cluster:
    large_df['pred_cluster'] = pred_clusters

    #Must calculate distance to cluster center for each of the dimensions:
    centroids = my_cluster.cluster_centers_

    num_dimensions = len(centroids[0])
    #print('Number of dimensions:', num_dimensions)

    large_df['dist_clustercenter'] = 'lol'

    cluster_distances = []
    #For each cluster
    for i in range(len(centroids)):
        rows = np.where(large_df['pred_cluster'] == i)[0].tolist()
        #For each row belonging to the relevant cluster:
        for _row in rows:
            distance = 0
            #Calculate the distance to cluster center for all dimensions
            for _dim in range(num_dimensions):
                distance += np.sqrt((large_df.iloc[_row, _dim] - centroids[i][_dim])**2)
            cluster_distances.append(distance)
            #Assign the distance value to corresponding row in df:
            large_df.loc[_row, 'dist_clustercenter'] = distance
    
#Pick unlabelled images in each cluster and
#label them according to labelled images in cluster if top 10 belongs to same label
    max_labeled = 0
    best_cluster = 0
    best_label = 'lol'
    for i in range(len(centroids)):
        num_obs = large_df.iloc[np.where(large_df['pred_cluster']==i)[0].tolist(),:].shape[0]
        print('Number of observations in cluster', i, ':' , num_obs)
        num_labeled = large_df.iloc[np.where( (large_df['pred_cluster']==i) & (large_df['Label']!= 'unlabeled'))[0].tolist(),:].shape[0]
        print('Number of labeled observations:', num_labeled)
        tot_polyps = 0
        tot_nonpolyps = 0
        #Select the cluster with highest number of labeled images (belonging to the same label)
        for _n in range(num_obs):
            if large_df.iloc[np.where(large_df['pred_cluster']==i)[0].tolist(),:].iloc[_n, -4] == 'polyps':
                tot_polyps += 1
            elif large_df.iloc[np.where(large_df['pred_cluster']==i)[0].tolist(),:].iloc[_n, -4] == 'nonpolyps':
                tot_nonpolyps += 1
        print('Total polyps in cluster:', tot_polyps)
        print('Total nonpolyps in cluster:', tot_nonpolyps)
        if (tot_polyps > max_labeled) and (tot_nonpolyps==0) and (num_labeled < num_obs):
            max_labeled = tot_polyps
            best_cluster = i
            best_label = 'polyps'
        elif (tot_nonpolyps > max_labeled) and (tot_polyps==0) and (num_labeled < num_obs):
            max_labeled = tot_nonpolyps
            best_cluster = i
            best_label = 'nonpolyps'
    print('Original number of unlabeled images:', len(large_df.iloc[np.where(large_df['Label']=='unlabeled')[0].tolist(), :]))
    
    print('Best cluster:', best_cluster)
    print('...With', max_labeled, 'labeled images inside')
    print('Type of label:', best_label)
    if best_label != 'lol':
        large_df['Label'] = np.where( (large_df['pred_cluster']==best_cluster) & (large_df['Label']== 'unlabeled'), best_label, large_df['Label'])
    print('Number of unlabeled images:', len(large_df.iloc[np.where(large_df['Label']=='unlabeled')[0].tolist(), :]))
    
    return large_df


run_clusterloop(large_df, num_clusters=50)

#Run clusterloop until at least 1000 labeled images in df: 
while large_df.iloc[np.where(large_df['Label']!='unlabeled')[0].tolist(), :].shape[0] < 1000:
    #Remove pred_clusters and dist_clustercenter from large_df before new round of clustering:
    large_df.drop(['pred_cluster', 'dist_clustercenter'], axis = 1, inplace = True)
    run_clusterloop(large_df, num_clusters=50)

#When 1000 labeled images or more, select 1000 of the labeled images:
labeled_df = large_df.iloc[np.where(large_df['Label']!= 'unlabeled')[0].tolist(),:].iloc[:1000,:]
print('Number of polyp observations:', len(labeled_df.iloc[np.where(labeled_df['Label']=='polyps')[0].tolist(), :]))
print('Number of nonpolyp observations:', len(labeled_df.iloc[np.where(labeled_df['Label']=='nonpolyps')[0].tolist(), :]))


#Copy labeled images to cnn_training_images folder
polyp_target_dir = 'tiles_1000_cnn_training_50clusters/polyp'
nonpolyp_target_dir = 'tiles_1000_cnn_training_50clusters/non_polyp'

source_path = 'training_tiles/unlabeled'
print('Shape',labeled_df.iloc[200:,:].shape[0])
#Since 200 first images are already labeled and in another source directory...
for _i in range(labeled_df.iloc[200:,:].shape[0]):
    image_file = labeled_df.iloc[200 + _i, -3]
    this_source_path = os.path.join(source_path, image_file)
    #source_path = labeled_df.iloc[200 + _i, -5]
    if labeled_df.iloc[200 + _i, -4] == 'polyps':
        shutil.copy(this_source_path, polyp_target_dir)
    else:
        shutil.copy(this_source_path, nonpolyp_target_dir)


