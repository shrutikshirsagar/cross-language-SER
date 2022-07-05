import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import argparse
import numpy as np
import tensorflow as tf
from sklearn.base import check_array
from cvxopt import solvers, matrix
import scipy.linalg as sla
import pandas as pd
import glob
import os
import fnmatch
from sklearn.decomposition import PCA

def pairwise_X(X, Y):
    X2 = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), [1, tf.shape(Y)[0]])
    Y2 = tf.tile(tf.reduce_sum(tf.square(Y), axis=1, keepdims=True), [1, tf.shape(X)[0]])
    XY = tf.matmul(X, tf.transpose(Y))
    return X2 + tf.transpose(Y2) - 2*XY


def _get_optim_function(Xs, Xt, kernel="linear", gamma=1., degree=2, coef=1.):
    
    n = len(Xs)
    m = len(Xt)
    p = Xs.shape[1]
    
    Lxx = tf.ones((n,n), dtype=tf.float64) * (1./(n**2))
    Lxy = tf.ones((n,m), dtype=tf.float64) * (-1./(n*m))
    Lyy = tf.ones((m,m), dtype=tf.float64) * (1./(m**2))
    Lyx = tf.ones((m,n), dtype=tf.float64) * (-1./(n*m))

    L = tf.concat((Lxx, Lxy), axis=1)
    L = tf.concat((L, tf.concat((Lyx, Lyy), axis=1)), axis=0)
    
    if kernel == "linear":
        
        @tf.function
        def func(W):
            Kxx = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xs))
            Kyy = tf.matmul(tf.matmul(Xt, tf.linalg.diag(W**1)), tf.transpose(Xt))
            Kxy = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xt))

            K = tf.concat((Kxx, Kxy), axis=1)
            K = tf.concat((K, tf.concat((Kyy, tf.transpose(Kxy)), axis=1)), axis=0)

            f = -tf.linalg.trace(tf.matmul(K, L))
            Df = tf.gradients(f, W)
            H = tf.hessians(f, W)
            return f, Df, H
        
    elif kernel == "rbf":
        
        @tf.function
        def func(W):
            Kxx = pairwise_X(tf.matmul(Xs, tf.linalg.diag(W**1)), Xs)
            Kyy = pairwise_X(tf.matmul(Xt, tf.linalg.diag(W**1)), Xt)
            Kxy = pairwise_X(tf.matmul(Xs, tf.linalg.diag(W**1)), Xt)

            K = tf.concat((Kxx, Kxy), axis=1)
            K = tf.concat((K, tf.concat((Kyy, tf.transpose(Kxy)), axis=1)), axis=0)
            K = tf.exp(-gamma * K)

            f = -tf.linalg.trace(tf.matmul(K, L))
            Df = tf.gradients(f, W)
            H = tf.hessians(f, W)
            return f, Df, H
        
    elif kernel == "poly":
        
        @tf.function
        def func(W):
            Kxx = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xs))
            Kyy = tf.matmul(tf.matmul(Xt, tf.linalg.diag(W**1)), tf.transpose(Xt))
            Kxy = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xt))

            K = tf.concat((Kxx, Kxy), axis=1)
            K = tf.concat((K, tf.concat((Kyy, tf.transpose(Kxy)), axis=1)), axis=0)
            K = (gamma * K + coef)**degree

            f = -tf.linalg.trace(tf.matmul(K, L))
            Df = tf.gradients(f, W)
            H = tf.hessians(f, W)
            return f, Df, H
        
    else:
        raise ValueError("kernel param should be in ['linear', 'rbf', 'poly']")
        
    return func


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
    
    threshold="auto"
    kernel="linear"
    gamma=1.
    degree=2
    coef=1.
    verbose=1
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
    n_components = DS.shape[1]
    print('number of PCA components', n_components)
    
    Xs = DS
    Xt = DT
    
    n = len(Xs)
    m = len(Xt)
    p = Xs.shape[1]

    optim_func = _get_optim_function(tf.identity(Xs),tf.identity(Xt), kernel, gamma, degree, coef)

    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (p,1))
        x = tf.identity(np.array(x).ravel())
        f, Df, H = optim_func(x)
        f = f.numpy()
        Df = Df[0].numpy().reshape(1, -1)
        H = H[0].numpy()
        if z is None: return matrix(f), matrix(Df)
        return matrix(f), matrix(Df), matrix(H)

    linear_const_G = -np.eye(p)
    squared_constraint_G = np.concatenate((np.zeros((1, p)), -np.eye(p)), axis=0)

    linear_const_h = np.zeros(p)
    squared_constraint_h = np.concatenate((np.ones(1), np.zeros(p)))

    G = matrix(np.concatenate((linear_const_G, squared_constraint_G)))
    h = matrix(np.concatenate((linear_const_h, squared_constraint_h)))
    dims = {'l': p, 'q': [p+1], 's':  []}

    solvers.options["show_progress"] = bool(verbose)

    sol = solvers.cp(F, G, h, dims)

    W = np.array(sol["x"]).ravel()

    selected_features_ = np.zeros(p, dtype=bool)

    if threshold == "auto":
        args = np.argsort(W).ravel()
        max_diff_arg = np.argmax(np.diff(W[args]))
        threshold = W[args[max_diff_arg]]
        selected_features_[W<=threshold] = 1
    else:
        selected_features_[W<=threshold] = 1

    if np.sum(selected_features_) == 0:
        raise Exception("No features selected")

    features_scores_ = W
   
    
    #########################################################
    # Compute target domain statistics

    for filename2 in target_data_path:
        DS1 = pd.read_csv(os.path.join(feat_path, filename2), sep=',',header=None).values

        names, times, DS1 = DS1[:,0:1], DS1[:,1:2], DS1[:,2:].astype(float)
       

        DT_t = DS1[:, selected_features_]

        df1=pd.DataFrame(np.concatenate([names, times, DT_t],1))
        df1.to_csv(out_path1+filename2, index = False, header=None)

    #########################################################
    # Iterate over source domain files and perform adaptation

    for filename1 in source_data_path:

        DS = pd.read_csv(os.path.join(feat_path, filename1), sep=',',header=None).values

        names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)

       

        DS_t = DS[:, selected_features_]

        df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
        df.to_csv(out_path1+filename1, index = False, header=None)
