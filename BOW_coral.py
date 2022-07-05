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

if __name__ == '__main__':
##### MSF features
    feat_path = "../eGEMAPS_BOW/"
    out_path1 = "../eGEMAPS_BOW_coral/"
    target_data_path = fnmatch.filter(os.listdir(feat_path), 'Devel_*')
    source_data_path = fnmatch.filter(os.listdir(feat_path), 'Train_*')
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
        if filename.startswith(('Devel')):
            data = pd.read_csv(os.path.join(feat_path, filename), sep=';',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)



    CT = np.cov(DT, rowvar=False) + np.eye(DT.shape[1])
    for filename1 in source_data_path:

        DS = pd.read_csv(os.path.join(feat_path, filename1), sep=';',header=None).values

        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
        CS = np.cov(DS, rowvar=False) + np.eye(DS.shape[1])
        DS = DS.dot( np.linalg.inv( sla.sqrtm(CS) ) )



        DS_t = DS.dot( sla.sqrtm( CT ) )

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+filename1, index = False, header=None)


    for filename in target_data_path:
        shutil.copy(os.path.join(feat_path, filename), os.path.join(out_path1, filename))


    feat_path = "../MSF_BOW/"
    out_path1 = "../MSF_BOW_coral/"
    target_data_path = fnmatch.filter(os.listdir(feat_path), 'Devel_*')
    source_data_path = fnmatch.filter(os.listdir(feat_path), 'Train_*')
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
        if filename.startswith(('Devel')):
            data = pd.read_csv(os.path.join(feat_path, filename), sep=';',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)



    CT = np.cov(DT, rowvar=False) + np.eye(DT.shape[1])
    for filename1 in source_data_path:

        DS = pd.read_csv(os.path.join(feat_path, filename1), sep=';',header=None).values

        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)

        ## Whitening

        CS = np.cov(DS, rowvar=False) + np.eye(DS.shape[1])
        DS = DS.dot( np.linalg.inv( sla.sqrtm(CS) ) )

        ## Coloring

        DS_t = DS.dot( sla.sqrtm( CT ) )

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+filename1, index = False, header=None)
    for filename in target_data_path:
        shutil.copy(os.path.join(feat_path, filename), os.path.join(out_path1, filename))
