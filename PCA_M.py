import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import argparse
import scipy.linalg as sla
import pandas as pd
import glob
import fnmatch
import os
import shutil
from sklearn.decomposition import PCA

if __name__ == '__main__':
##### MSF features
    feat_path = "../eGEMAPS_avg_std/"
    out_path1 = "../eGEMAPS_avg_std_PCA/"
    target_data_path = fnmatch.filter(os.listdir(feat_path), 'Train_*')
    source_data_path = fnmatch.filter(os.listdir(feat_path), 'Devel_*')
    print(len(source_data_path), len(target_data_path))
    source_data_path.sort()
    target_data_path.sort()

    n_target_files = len(target_data_path)
    n_source_files = len(source_data_path)
    if not os.path.exists(out_path1):
        os.mkdir(out_path1)
    
    
   # standadization

    DT = None

    for filename in target_data_path:
        if filename.startswith(('Train')):
            data = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)



    pca = PCA(n_components=23)
    pca.fit(DT)
    for filename1 in source_data_path:

        DS = pd.read_csv(os.path.join(feat_path, filename1), sep=',',header=None).values

        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
        


        DS_t = pca.transform(DS)

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+filename1, index = False, header=None)


    for filename2 in target_data_path:
        DT1 = pd.read_csv(os.path.join(feat_path, filename2), sep=',',header=None).values

        names, times, DT1 = DT1[:,0:1], DT1[:,1:2], DT1[:,2:].astype(float)
        


        DS_t = pca.transform(DT1)

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+filename2, index = False, header=None)


    feat_path = "../MSF_std/"
    out_path1 = "../MSF_std_PCA/"
    target_data_path = fnmatch.filter(os.listdir(feat_path), 'Train_*')
    source_data_path = fnmatch.filter(os.listdir(feat_path), 'Devel_*')
    print(len(source_data_path), len(target_data_path))
    source_data_path.sort()
    target_data_path.sort()

    n_target_files = len(target_data_path)
    n_source_files = len(source_data_path)
    if not os.path.exists(out_path1):
        os.mkdir(out_path1)
    
    
   # standadization

    DT = None

    for filename in target_data_path:
        if filename.startswith(('Train')):
            data = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)



    pca = PCA(n_components=20)
    pca.fit(DT)
    for filename1 in source_data_path:

        DS = pd.read_csv(os.path.join(feat_path, filename1), sep=',',header=None).values

        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
        


        DS_t = pca.transform(DS)

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+filename1, index = False, header=None)


    for filename2 in target_data_path:
        DT1 = pd.read_csv(os.path.join(feat_path, filename2), sep=',',header=None).values

        names, times, DT1 = DT1[:,0:1], DT1[:,1:2], DT1[:,2:].astype(float)
        


        DS_t = pca.transform(DT1)

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+filename2, index = False, header=None)

