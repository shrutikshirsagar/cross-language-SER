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


from scipy.spatial.distance import cdist
def is_pos_def(X):
    """ Check for positive definiteness.
    """
    
    return np.all(np.linalg.eigvals(X) > 0)
def kernel(X1, X2=None, kernel_type='rbf', sigma=1.0, degree=1):
    """ Compute the kernel for a given X1 and X2.
    Parameters
    ----------
    X1 : np.array of shape (n_samples, n_features)
        Input instances X1.
    X2 : np.array of shape (n_samples, n_features)
        Input instances of X2.
    kernel_type : str (default='rbf')
        Different kernel types in:
        [rbf, linear, sigmoid, polynomial, primal, sam]
    sigma : float (default=1.0)
        Sigma (bandwidth) of the rbf kernel.
        Sigma parameter of the sam kernel.
    degree : int (default=1)
        Degree of the polynomial kernel.
    Returns
    -------
    K : np.array of shape (n_samples, n_samples)
        Kernel computed on X1 and X2.
    """

    # make copy
    if X2 is None:
        X2 = X1.copy()

    # data shape
    n1, nf1 = X1.shape
    n2, nf2 = X2.shape

    # dimension check
    if not(nf1 == nf2):
        raise ValueError('Dimensions of kernel input matrices should be equal')

    # compute the kernels
    if kernel_type.lower() == 'primal':
        K = X1

    elif kernel_type.lower() == 'linear':
        K = np.dot(X1, X2.T)

    elif kernel_type.lower() == 'rbf':
        K = np.exp(-cdist(X1, X2) / (2.0 * (sigma ** 2)))

    elif kernel_type.lower() == 'sigmoid':
        K = 1.0 / (np.exp(np.dot(X1, X2.T)) + 1.0)

    elif kernel_type.lower() == 'polynomial':
        K = np.power(np.dot(X1, X2.T) + 1.0, degree)

    elif kernel_type.lower() == 'sam':
        D = np.dot(X1, X2.T)
        D_flat = D.ravel()
        acos_func = np.vectorize(math.acos)
        D_flat_acos = acos_func(D_flat)
        D = D_flat_acos.reshape(D.shape)

        # kernel
        K = np.exp(np.power(D, 2) / (2.0 * (sigma ** 2)))

    else:
        raise ValueError(kernel_type,
            'not in [rbf, linear, sigmoid, polynomial, primal, sam]')
    
    # check output dimensions
    assert K.shape == (n1, n2), 'kernel matrix has wrong dimensions'

    return K

if __name__ == '__main__':
    feat_path = "//media/amrgaballah/Backup_Plus/exp_J3/cross-language/F_train_G_test/MSF_std/"
    out_path1 = "//media/amrgaballah/Backup_Plus/exp_J3/cross-language/F_train_G_test/MSF_std_coral/"
    target_data_path = fnmatch.filter(os.listdir(feat_path), 'Devel_*')
    source_data_path = fnmatch.filter(os.listdir(feat_path), 'Train_*')
    print(len(source_data_path), len(target_data_path))
    source_data_path.sort()
    target_data_path.sort()

    n_target_files = len(target_data_path)
    n_source_files = len(source_data_path)
    if not os.path.exists(out_path1):
        os.mkdir(out_path1)
    
    
    mu=0.1
    n_components=39
    kernel_type='linear'
    sigma=1.0
    degree=2
    tol=1e-8
    #########################################################
    # Read target data
    
   # standadization

    DT = None

    for filename in target_data_path:
        if filename.startswith(('Devel')):
            data = pd.read_csv(os.path.join(feat_path, filename), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DT is not None:
                DT = np.concatenate([DT, data], 0)
                print(DT.shape)
            else:
                DT = data
                print(DT.shape)
   # standadization

    DS = None

    for filename1 in source_data_path:
        if filename.startswith(('Devel')):
            data = pd.read_csv(os.path.join(feat_path, filename1), sep=',',header=None).values[:,2:].astype(float)
            print(data.shape)
            if DS is not None:
                DS = np.concatenate([DS, data], 0)
                print(DS.shape)
            else:
                DS = data
                print(DS.shape)
    Xs = DS
    Xt = DT
    ns, nfs = Xs.shape
    nt, nft = Xt.shape
    X_ = np.concatenate((Xs, Xt), axis=0)
    K = kernel(X_, X_, kernel_type=kernel_type, sigma=sigma, degree=degree)

    # kernel matrix should be postive definite
    # adapted from: https://github.com/wmkouw/libTLDA/blob/master/libtlda/tca.py
    if not is_pos_def(K):
        print('Warning: covariate matrices not PSD.')

        regct = -6
        while not is_pos_def(K):
            print('Adding regularization:', 10 ** regct)

            # Add regularization
            K += np.eye(ns + nt) * (10.0 ** regct)

            # Increment regularization counter
            regct += 1

    # coefficient matrix L
    L = np.vstack((np.hstack((np.ones((ns, ns)) / ns ** 2, -1.0 * np.ones((ns, nt)) / (ns * nt))), np.hstack((-1.0 * np.ones((nt, ns)) / (ns * nt), np.ones((nt, nt)) / (nt ** 2)))))

    # centering matrix H
    H = np.eye(ns + nt) - np.ones((ns + nt, ns + nt)) / float(ns + nt)

    # matrix Lagrangian objective function: (I + mu*K*L*K)^{-1}*K*H*K
    J = np.dot(np.linalg.inv(np.eye(ns + nt) + mu * np.dot(np.dot(K, L), K)), np.dot(np.dot(K, H), K))

    # eigenvector decomposition as solution to trace minimization
    _, C = eigs(J, k=n_components)

    # transformation/embedding matrix
    C_ = np.real(C)

    # transform the source data
    Xs_trans_ = np.dot(K[:ns, :], C_)
    print('Xs_trans_', Xs_trans_.shape)
    Xt_trans_ = np.dot(K[ns:, :], C_)
    print('Xt_trans_', Xt_trans_.shape)
    Ixs_trans_ = np.arange(0, ns, 1)
        
   
    #########################################################
    # Iterate over source domain files and perform adaptation
    i = 0
    j = 0
    while j < Xs_trans_.shape[0]:
        for filename2 in source_data_path:

            DS = pd.read_csv(os.path.join(feat_path, filename1), sep=',',header=None).values

            names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)
      
            j = i+ DS.shape[0]
            DS_t = Xs_trans_[i:j, :]
            i = i+ DS.shape[0]
            print(DS_t.shape)
            df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
            df.to_csv(out_path1+filename2, index = False, header=None)    
