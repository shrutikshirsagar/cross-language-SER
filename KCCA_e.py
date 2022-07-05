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
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
if __name__ == '__main__':
    feat_path = "../eGEMAPS_avg_std/"
    out_path1 = "../eGEMAPS_avg_std_KCCA/"
    target_data_path = fnmatch.filter(os.listdir(feat_path), 'Train_*')
    source_data_path = fnmatch.filter(os.listdir(feat_path), 'Devel_*')
    print(len(source_data_path), len(target_data_path))
    source_data_path.sort()
    target_data_path.sort()

    n_target_files = len(target_data_path)
    n_source_files = len(source_data_path)
    if not os.path.exists(out_path1):
        os.mkdir(out_path1)
    
    
   
    #########################################################
    # Read target data

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
                
   # standadization
    print('DTshape', DT.shape)
    DS = None

    for filename1 in source_data_path:
        if filename1.startswith(('Devel')):
            data = pd.read_csv(os.path.join(feat_path, filename1), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DS is not None:
                DS = np.concatenate([DS, data], 0)
                print(DS.shape)
            else:
                DS = data
               
    
    print('DSshape', DS.shape)
    n_components = DS.shape[1]
    print('number of PCA components', n_components)
    pca_src_ = PCA(n_components)
    pca_tgt_ = PCA(n_components)
        
    pca_src_.fit(DS)
    pca_tgt_.fit(DT)
    X_x = pca_src_.transform(DS)
    print('X_x', X_x.shape)
    X_y = pca_src_.transform(DT) 
    print('X_y', X_y.shape)
    Y_x = pca_tgt_.transform(DS)
    print('Y_x', Y_x.shape)
    Y_y = pca_tgt_.transform(DT)
    print('Y_y', Y_y.shape)
    map_x = np.concatenate([X_x, Y_y], axis=0)
    map_y = np.concatenate([X_y, Y_x], axis=0)
        
    print(map_x.shape)
    print(map_y.shape) 
    cca = CCA (n_components = 10)
    cca.fit(map_x, map_y)

    # Compute target domain statistics

    for target_file in target_data_path:

        DT4 = pd.read_csv(os.path.join(feat_path, target_file), sep=',',header=None).values

        names, times, DT4 = DT4[:,0:1], DT4[:,1:2], DT4[:,2:].astype(float)

       

        DT_t4 = cca.transform(DT4)

        df1=pd.DataFrame(np.concatenate([names, times, DT_t4],1))
        df1.to_csv(out_path1+target_file, index = False, header=None)

    #########################################################
    # Iterate over source domain files and perform adaptation

    for source_file in source_data_path:

        DS4 = pd.read_csv(os.path.join(feat_path,source_file), sep=',',header=None).values

        names, times, DS4 = DS4[:,0:1], DS4[:,1:2], DS4[:,2:].astype(float)

       

        DS_t4 = cca.transform(DS4)

        df=pd.DataFrame(np.concatenate([names, times, DS_t4],1))
        df.to_csv(out_path1+source_file, index = False, header=None) 

    
