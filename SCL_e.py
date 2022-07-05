import numpy as np
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from utils import mmd_coef, base_init
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import argparse
import scipy.linalg as sla
import pandas as pd
import glob
import os
from scipy.sparse.linalg import eigs
from sklearn.decomposition import PCA
import fnmatch
import math
import numpy as np
import scipy.io
import scipy.linalg
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import scipy as sp
import sklearn 
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

class SCL(object):
    
    
    

    def fit(Xs, Xt):
        '''
        find pivot features and transfer the Xs and Xt
        Param Xs: source data
        Param Xt: target data
        output Xs_new: new source data features
        output Xt_new: new target data features
        output W: transform matrix
        '''
        _, ds = Xs.shape
        _, dt = Xt.shape
        assert ds == dt
        X = np.concatenate((Xs, Xt), axis=0)
        ix = np.argsort(np.sum(X, axis=0))
        ix = ix[::-1][:num_pivots]
        pivots = (X[:, ix]>0).astype('float')
        p = np.zeros((ds, num_pivots),dtype = 'complex_')
        # train for the classifers 
        for i in range(num_pivots):
            clf = linear_model.SGDClassifier(loss="modified_huber", alpha=l2)
            clf.fit(X, pivots[:, i])
            p[:, i] = clf.coef_
        _, W = np.linalg.eig(np.cov(p))
        W = W[:, :num_pivots]
        W = W
   

        return W

    def transform(X):
        '''
        transform the origianl data by add new features
        Param X: original data
        output x_new: X with new features
        '''
        X_new = np.concatenate((np.dot(X, W),X), axis=1)
        return X_new
# from sklearn.preprocessing import StandardScaler
# =============================================================================
# Implementation of three transfer learning methods:
#   1. Transfer Component Analysis: TCA
#   2. Joint Distribution Adaptation: JDA
#   3. Balanced Distribution Adaptation: BDA
# Ref:
# [1] S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via
# Transfer Component Analysis," in IEEE Transactions on Neural Networks,
# vol. 22, no. 2, pp. 199-210, Feb. 2011.
# [2] Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, Philip S. Yu,
# Transfer Feature Learning with Joint Distribution Adaptation, IEEE 
# International Conference on Computer Vision (ICCV), 2013.
# [3] Wang, J., Chen, Y., Hao, S., Feng, W. and Shen, Z., 2017, November. Balanced
# distribution adaptation for transfer learning. In Data Mining (ICDM), 2017
# IEEE International Conference on (pp. 1129-1134). IEEE.
# =============================================================================
if __name__ == '__main__':
    feat_path = "../eGEMAPS_avg_std/"
    out_path1 = "../eGEMAPS_avg_std_SCL/"
    target_data_path = fnmatch.filter(os.listdir(feat_path), 'Train_*')
    source_data_path = fnmatch.filter(os.listdir(feat_path), 'Devel_*')
    print(len(source_data_path), len(target_data_path))
    source_data_path.sort()
    target_data_path.sort()

    n_target_files = len(target_data_path)
    n_source_files = len(source_data_path)
    if not os.path.exists(out_path1):
        os.mkdir(out_path1)
    
    l2=1.0
    num_pivots=10 
   
    W = 0
    mu=0.1
    n_components=10
    kernel_type='linear'
    sigma=1.0
    degree=2
    tol=1e-8
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
                print('DT.shape', DT.shape)
   # standadization
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
                print('DS.shape', DS.shape)
    Xs = DS
    Xt = DT
    print(Xs.shape, Xt.shape)   

    W = SCL.fit(Xs, Xt)
        # Compute target domain statistics

    for target_file in target_data_path:

        DT4 = pd.read_csv(os.path.join(feat_path, target_file), sep=',',header=None).values

        names, times, DT4 = DT4[:,0:1], DT4[:,1:2], DT4[:,2:].astype(float)

        DT_t4 = np.concatenate((np.dot(DT4, W).real, DT4), axis=1)

        #DT_t4 = cca.transform(DT4)
        #DT_t4 = DT_t4.real
        df1=pd.DataFrame(np.concatenate([names, times, DT_t4],1))
        df1.to_csv(out_path1+target_file, index = False, header=None)

    #########################################################
    # Iterate over source domain files and perform adaptation

    for source_file in source_data_path:

        DS4 = pd.read_csv(os.path.join(feat_path,source_file), sep=',',header=None).values

        names, times, DS4 = DS4[:,0:1], DS4[:,1:2], DS4[:,2:].astype(float)

        

        DS_t4 =  np.concatenate((np.dot(DS4, W).real, DS4), axis=1)
        #DS_t4 = DT_t4.real
        df=pd.DataFrame(np.concatenate([names, times, DS_t4],1))
        df.to_csv(out_path1+source_file, index = False, header=None) 

    
