import pandas as pd
import numpy as np




def validate_input(X, classtfidf=False):
    is_embedding = False
    
    if isinstance(X, list):
        X = np.array(X)
    elif hasattr(X, 'values'):
        X = X.values
        if X.ndim > 1:
            raise ValueError('Only Pandas Series accepted. Please pass in only the relevant column of the dataframe.')                
        X = np.array(X)
    elif isinstance(X, np.ndarray):
        pass
    elif isinstance(X, pd.arrays.ArrowStringArray):
        X = np.array(X)
    else:
        raise ValueError('Only list, NumPy arrays or Pandas Series types accepted.')
    
    if not isinstance(X[0], str):
        is_embedding = True
        if classtfidf:
            raise ValueError('ClassTfidf expects a list or array of string values.')
        
    return X, is_embedding



def validate_metric(metric):
    metrics = ['euclidean', 'manhattan', 'cosine', 'haversine', 'precomputed']
    if metric not in metrics:
        raise ValueError(f'Metric must be one of {metrics}. For precomputed, you must provide a square matrix of pairwise distances.')

    return metric

def validate_tree_metric(metric):
    metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean']
    if metric not in metrics:
        raise ValueError(f'Metric must me one of {metrics}.')
    
    return metric


class Encoder():
    
    def __init__(self, encoder):
        """
        Encoder passed to the EHDBSCAN must subclass this Encoder class.
        """
        pass
        
    def encode(self, X):
        """
        All subclasses must implement the encode method.

        Parameters
        ----------
        X : {array-like} of shape (n_samples,)
        Text input acceptable by EHDBSCAN.

        Returns
        -------
        {array-like} of shape (n_samples, n_features)
        Must output array of embeddings of specified shape.
        """
        
        raise NotImplementedError('Subclass must implement the encode method and return an array-like of shape (n_samples, n_features).')


class Reducer():
    def __init__(self):
        """
        Dimension reduction method passed to EHDBSCAN must subclass this Reducer class.
        """
        pass

    def fit(self, X):
        """
        All subclasses must implement this method.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
        Fit the input embeddings on the reduction method.

        Expects encoded embeddings input with numeric dtype.

        Returns
        -------
        self : object
        Expects return self object, but not required.
        
        """
        
        raise NotImplementedError('Subclass must implement the fit method.')

    def transform(self, X):
        """
        All subclasses must implement this method and return an array-like object
        of shape (n_samples, n_features).

        EHDBSCAN will call fit and transform separately. A fit_transform may be
        written for personal use but is not required.

        Parameters
        ---------
        X : {array-like} of shape (n_samples, n_features)
        Transform the encoded embeddings with the fitted reduction method.

        Expects encoded embeddings input with numeric dtype.

        Returns
        -------
        {array-like} of shape (n_samples, n_reduced_features)
        Method must return an array-like object that will be used for clustering.

        These embeddings will be stored in ClusterFeatures for approx_predict of
        EHDBSCAN.
        
        """
        
        raise NotImplementedError('Subclass must implement the transform method.')
        


