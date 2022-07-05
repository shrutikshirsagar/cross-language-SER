import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import argparse
import scipy.linalg as sla
import pandas as pd
import glob
import os
import fnmatch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import argparse
import scipy.linalg as sla
import pandas as pd
import glob
from sklearn import preprocessing
import numpy as np
import os

# Get all files

if __name__ == '__main__':
   
    feat_path = "../eGEMAPS_avg/"
    out_path1 = "../eGEMAPS_avg_std/"
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

    for filename in source_data_path:
        if filename.startswith(('Train')):
            data = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)
    # standarization
    X_scaled = preprocessing.scale(DT)
    print(X_scaled)
    scaler = preprocessing.StandardScaler().fit(DT)
    print(scaler)
    DT = scaler.transform(DT)
    print(DT.shape)


    for filename in target_data_path:
        DS = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values
        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
        DS_n = scaler.transform(DS)
        print(DS_n.shape)
        df=pd.DataFrame(np.concatenate([names, times, DS_n],1))
        df.to_csv(out_path1+filename, index = False, header=None)

    for filename in source_data_path:
        DT1 = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values
        names, times, DT1 = DT1[:,0:1], DT1[:,1:2], DT1[:,2:].astype(float)
        DS_n1 = scaler.transform(DT1)
        print(DS_n1.shape)
        df1=pd.DataFrame(np.concatenate([names, times, DS_n1],1))
        df1.to_csv(out_path1+filename, index = False, header=None)

    
    #########################################################
    # SA

    #feat_path = "//media/amrgaballah/Backup_Plus/exp_J3/cross-language/F_train_G_test/eGEMAPS_avg/"
    feat_path1= "../eGEMAPS_avg_std/"
    out_path2 = "../eGEMAPS_avg_DA_SA/"
    target_data_path1 = fnmatch.filter(os.listdir(feat_path1), 'Devel_*')
    source_data_path1 = fnmatch.filter(os.listdir(feat_path1), 'Train_*')
    #print(len(source_data_path), len(target_data_path))
    source_data_path1.sort()
    target_data_path1.sort()

    n_target_files = len(target_data_path1)
    n_source_files = len(source_data_path1)
    if not os.path.exists(out_path2):
        os.mkdir(out_path2)
    
    DT = None

    for filename in source_data_path1:
        if filename.startswith(('Train')):
            data = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)
    DS = None

   

    for filename in target_data_path1:
        if filename.startswith(('Devel')):
            data = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DS is not None:
                DS = np.concatenate([DS, data], 0)
                print(DS.shape)
            else:
                DS = data
                print(DS.shape)
    n_components = DS.shape[1]
    print('number of PCA components', n_components)
    pca_src_ = PCA(n_components)
    pca_tgt_ = PCA(n_components)
        
    pca_src_.fit(DS)
    pca_tgt_.fit(DT)
        
    M_  = pca_src_.components_.dot(pca_tgt_.components_.transpose())
        
       
    #########################################################
    # Compute target domain statistics

    for filename in target_data_path1:
        DT = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values
        names, times, DT = DT[:,0:1], DT[:,1:2], DT[:,2:].astype(float)

       

        DT_t = pca_tgt_.transform(DT)

        df1=pd.DataFrame(np.concatenate([names, times, DT_t],1))
        df1.to_csv(out_path2+filename, index = False, header=None)

    #########################################################
    # Iterate over source domain files and perform adaptation

    for filename in source_data_path1:
        DS = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values
        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
       

        DS_t = pca_src_.transform(DS).dot(M_)

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path2+filename, index = False, header=None)

##### MSF features
    feat_path = "../MSF/"
    out_path1 = "../MSF_std/"
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

    for filename in source_data_path:
        if filename.startswith(('Train')):
            data = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)
    # standarization
    X_scaled = preprocessing.scale(DT)
    print(X_scaled)
    scaler = preprocessing.StandardScaler().fit(DT)
    print(scaler)
    DT = scaler.transform(DT)
    print(DT.shape)


    for filename in target_data_path:
        DS = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values
        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
        DS_n = scaler.transform(DS)
        print(DS_n.shape)
        df=pd.DataFrame(np.concatenate([names, times, DS_n],1))
        df.to_csv(out_path1+filename, index = False, header=None)

    for filename in source_data_path:
        DT1 = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values
        names, times, DT1 = DT1[:,0:1], DT1[:,1:2], DT1[:,2:].astype(float)
        DS_n1 = scaler.transform(DT1)
        print(DS_n1.shape)
        df1=pd.DataFrame(np.concatenate([names, times, DS_n1],1))
        df1.to_csv(out_path1+filename, index = False, header=None)

    
    #########################################################
    # SA

    #feat_path = "//media/amrgaballah/Backup_Plus/exp_J3/cross-language/F_train_G_test/eGEMAPS_avg/"
    feat_path1= "../MSF_std/"
    out_path2 = "../MSA_DA_SA/"
    target_data_path1 = fnmatch.filter(os.listdir(feat_path1), 'Devel_*')
    source_data_path1 = fnmatch.filter(os.listdir(feat_path1), 'Train_*')
    #print(len(source_data_path), len(target_data_path))
    source_data_path1.sort()
    target_data_path1.sort()

    n_target_files = len(target_data_path1)
    n_source_files = len(source_data_path1)
    if not os.path.exists(out_path2):
        os.mkdir(out_path2)
    
    DT = None

    for filename in source_data_path1:
        if filename.startswith(('Train')):
            data = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)
    DS = None

   

    for filename in target_data_path1:
        if filename.startswith(('Devel')):
            data = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DS is not None:
                DS = np.concatenate([DS, data], 0)
                print(DS.shape)
            else:
                DS = data
                print(DS.shape)
    n_components = DS.shape[1]
    print('number of PCA components', n_components)
    pca_src_ = PCA(n_components)
    pca_tgt_ = PCA(n_components)
        
    pca_src_.fit(DS)
    pca_tgt_.fit(DT)
        
    M_  = pca_src_.components_.dot(pca_tgt_.components_.transpose())
        
       
    #########################################################
    # Compute target domain statistics

    for filename in target_data_path1:
        DT = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values
        names, times, DT = DT[:,0:1], DT[:,1:2], DT[:,2:].astype(float)

       

        DT_t = pca_tgt_.transform(DT)

        df1=pd.DataFrame(np.concatenate([names, times, DT_t],1))
        df1.to_csv(out_path2+filename, index = False, header=None)

    #########################################################
    # Iterate over source domain files and perform adaptation

    for filename in source_data_path1:
        DS = pd.read_csv(os.path.join(feat_path1, filename), sep=',',header=None).values
        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
       

        DS_t = pca_src_.transform(DS).dot(M_)

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path2+filename, index = False, header=None)
