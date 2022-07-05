import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import argparse
import scipy.linalg as sla
import pandas as pd
import glob
import os
from sklearn.decomposition import PCA
if __name__ == '__main__':

    target_data_path ="///media/amrgaballah/Backup_Plus/exp_J3/cross-language/MFCC_exp/Germantrain_frenchtest/withoutDA/standarization/test/"
    out_path1 ="//media/amrgaballah/Backup_Plus/exp_J3/cross-language/MFCC_exp/Germantrain_frenchtest/withoutDA/standarization//Train_adap/"
    
    source_data_path ="//media/amrgaballah/Backup_Plus/exp_J3/cross-language/MFCC_exp/Germantrain_frenchtest/withoutDA/standarization/train/"
    source_domain_files = glob.glob(source_data_path+'*.csv')
    target_domain_files = glob.glob(target_data_path+'*.csv')
    n_target_files = 9
    n_source_files = 48
    if not os.path.exists(out_path1):
        os.mkdir(out_path1)
    
    
    source_domain_files = glob.glob(source_data_path+'*.csv')
    target_domain_files = glob.glob(target_data_path+'*.csv')
    
    #########################################################
    # Read target data

    DT = None

    for i in range(min(n_target_files, len(target_domain_files))):
        data = pd.read_csv(target_domain_files[i], sep=',',header=None).values[:,2:].astype(float)
        if DT is not None:
            DT = np.concatenate([DT, data], 0)
        else:
            DT = data

    DS = None

    for i in range(min(n_source_files, len(source_domain_files))):
        data1 = pd.read_csv(source_domain_files[i], sep=',',header=None).values[:,2:].astype(float)
        if DS is not None:
            DS = np.concatenate([DS, data1], 0)
        else:
            DS = data1
    n_components = DS.shape[1]
    print('number of PCA components', n_components)
    pca_src_ = PCA(n_components)
    pca_tgt_ = PCA(n_components)
        
    pca_src_.fit(DS)
    pca_tgt_.fit(DT)
        
    M_  = pca_src_.components_.dot(pca_tgt_.components_.transpose())
        
       
    #########################################################
    # Compute target domain statistics

    for target_file in target_domain_files:

        DT = pd.read_csv(target_file, sep=',',header=None).values

        names, times, DT = DT[:,0:1], DT[:,1:2], DT[:,2:].astype(float)

       

        DT_t = pca_tgt_.transform(DT)

        df1=pd.DataFrame(np.concatenate([names, times, DT_t],1))
        df1.to_csv(out_path1+target_file.split('/')[-1].split('.')[0]+'.csv', index = False, header=None)

    #########################################################
    # Iterate over source domain files and perform adaptation

    for source_file in source_domain_files:

        DS = pd.read_csv(source_file, sep=',',header=None).values

        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)

       

        DS_t = pca_src_.transform(DS).dot(M_)

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+source_file.split('/')[-1].split('.')[0]+'.csv', index = False, header=None)
