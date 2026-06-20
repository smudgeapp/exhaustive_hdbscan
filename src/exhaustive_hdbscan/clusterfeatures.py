import pandas as pd
import numpy as np
from typing import Any



class ClusterFeatures:

    """
    The data layer for the EHDBSCAN. It provides storage and easy access of
    clusters, inputs, distances and embeddings.

    The ClusterFeatures object is accessible as an attribute from the EHDBSCAN
    object after the fit is called. It is returned by the approx_predict method as
    a separate instance.

    It is not intended for populating manually. Storage are automatically encoded with
    designated name where names are sequentially automatically generated.

    Naming convention will assign names as {cluster type}_{iteration index}. For example
    second iteration will designate the generated labels as "parent_1" and the assigned child
    labels as "child_0". Here "parent_1" is the current iteration label and "child_0" is the
    child of the previous iteration "parent_0".

    Note that any child labels will have the same unique indexes as its parent indicating which
    parent that child belongs to.

    Attributes
    ----------
    parent_names : list
    List of all parent names in the order their labels and embeddings are stored.

    child_names : list
    List of all children names in the order their labels and embeddings are stored.
    
    """

    def __init__(self):
        self.parent_labels = np.array([])
        self.child_labels = np.array([])
        self.parent_names = []
        self.child_names = []
        self.pandas_data = pd.DataFrame()
        self.distances = 0
        self.embeddings = np.array([])
        self.reduce_embeddings = np.array([])
        self.input = np.array([])
        
        
    def generate_name(self, child=False):
        key_infer = 'parent_0'
        if len(self.parent_names) > 0:
            key_infer = f'parent_{len(self.parent_names)}'

        if child:
            key_infer = 'child_0'
            if len(self.child_names) > 0:
                key_infer = f'child_{len(self.child_names)}'
                
        return key_infer
    
    def _add_input(self, X):
        self.pandas_data['input'] = list(X)
        self.input = self.pandas_data['input'].to_numpy()
    
    def _add_parent(self, X):
        key_infer = self.generate_name()
        if len(self.parent_names) > 0:
            self.parent_labels = np.concatenate((self.parent_labels, X[:, np.newaxis]), axis=1)
        else:
            self.parent_labels = X[:, np.newaxis]
        
        self.pandas_data[key_infer] = list(X)
        self.parent_names.append(key_infer)
        return self.parent_names

    def _add_child(self, X):
        key_infer = self.generate_name(child=True)
        if len(self.child_names) > 0:
            self.child_labels = np.concatenate((self.child_labels, X[:, np.newaxis]), axis=1)
        else:
            self.child_labels = X[:, np.newaxis]        
        self.pandas_data[key_infer] = list(X)
        self.child_names.append(key_infer)
        return self.child_names

    def _add_distances(self, X):
        self.distances = np.ascontiguousarray(X)
    
    def _add_embeddings(self, X):
        self.embeddings = np.ascontiguousarray(X)

    def _add_reduce_embeddings(self, X):
        if len(self.parent_names) > 0:
            self.reduce_embeddings = np.concatenate((self.reduce_embeddings, X[np.newaxis]), axis=0)
        else:
            self.reduce_embeddings = X[np.newaxis]
        
    def get_parent(self, name: str = None, label: int = None, get_ids: Any = None):
        """
        Get the label of parent by name, label and index ID.

        If neither is specified, all labels are returned.

        Parameters
        ----------
        name : str
        Name of parent from parent_names.

        label : int
        Label from the parent where name must be specified.

        If label is specified without name, this method will throw an error.

        get_ids : {array-like} of shape (n_samples,)
        Give specific index IDs to be extracted from the parent labels.

        If no name or label is specified, it will return the IDs for all parent
        clusters.

        Returns
        -------
        ndarray, ndarray
        Returns an array of the labels and their respective IDs
        
        """
        parent_out = self.parent_labels
        select_ids = np.arange(0, parent_out.shape[0])
        idx = None
        
        if name is not None:
            try:
                idx = self.parent_names.index(name)
            except:
                raise ValueError('Parent name not found.')
            parent_out = self.parent_labels[:, idx]
            
        if label is not None:
            if idx is None:
                raise ValueError('When label is specified, specific parent name must be provided to retrieve the labels.')
            else:
                label_mask = (parent_out == label)
                select_ids = select_ids[label_mask]
                parent_out = parent_out[label_mask]
        
        if get_ids is not None:
            id_mask = np.isin(select_ids, get_ids)
            select_ids = select_ids[id_mask]
            parent_out = parent_out[id_mask]

        return parent_out, select_ids

    def get_child(self, name=None, label=None, get_ids=None):
        """
        Get the label of child by name, label and index ID.

        If neither is specified, all labels are returned.

        Parameters
        ----------
        name : str
        Name of child from child_names.

        label : int
        Label from the child where name must be specified.

        If label is specified without name, this method will throw an error.

        get_ids : {array-like} of shape (n_samples,)
        Give specific index IDs to be extracted from the child labels.

        If no name or label is specified, it will return the IDs for all child
        clusters.

        Returns
        -------
        ndarray, ndarray
        Returns an array of the labels and their respective IDs
        
        """
        child_out = self.child_labels
        select_ids = np.arange(0, child_out.shape[0])
        idx = None

        if name is not None:
            try:
                idx = self.child_names.index(name)
            except:
                raise ValueError('Child name not found.')
            child_out = self.child_labels[:, idx]
            
        if label is not None:
            if idx is None:
                raise ValueError('When label is specified, specific child name must be provided to retrieve the labels.')
            else:
                label_mask = (child_out == label)
                select_ids = select_ids[label_mask]
                child_out = child_out[label_mask]
        
        if get_ids is not None:
            id_mask = np.isin(select_ids, get_ids)
            select_ids = select_ids[id_mask]
            child_out = child_out[id_mask]

        return child_out, select_ids

    def get_input(self, name=None, label=None, get_ids=None):
        """
        Get the input text by parent or child name, label and index ID.

        If neither is specified, all input is returned.

        Parameters
        ----------
        name : str
        Name of parent or child from parent_names or child_names.

        label : int
        Label from the parent or child where name must be specified.

        If label is specified without name, this method will throw an error.

        get_ids : {array-like} of shape (n_samples,)
        Give specific index IDs to be extracted from the input.

        If no name or label is specified, it will return the IDs for all clusters.

        Returns
        -------
        ndarray, ndarray
        Returns an array of the input text and their respective IDs
        
        """
        in_out = self.input
        name_array = None
        idx = None
        select_ids = np.arange(0, self.input.shape[0])
        is_child = False

        if name is not None:
            try:
                idx = self.parent_names.index(name)
            except:
                try:
                    idx = self.child_names.index(name)
                    is_child = True
                except:
                    raise ValueError('Name not found in parent and child.')
            
            if is_child:
                name_array = self.child_labels[:, idx]
            else:
                name_array = self.parent_labels[:, idx]

            name_mask = (name_array >= 0)
            in_out = self.input[name_mask]
            name_array = name_array[name_mask]
            select_ids = select_ids[name_mask]

        if label is not None:
            if idx is None:
                raise ValueError('When label is specified, specific parent or child name must be provided to retrieve the values.')
            else:
                label_mask = (name_array == label)
                in_out = in_out[label_mask]
                select_ids = select_ids[label_mask]

        if get_ids is not None:
            id_mask = np.isin(select_ids, get_ids)
            select_ids = select_ids[id_mask]
            in_out = in_out[id_mask]

        return in_out, select_ids

    
    def get_distances(self, name=None, label=None, get_ids=None, row_ids=None, col_ids=None):
        """
        Get the pairwise distances by parent or child name, label and index ID.

        If neither is specified, all distances are returned.

        Parameters
        ----------
        name : str
        Name of parent or child from parent_names or child_names.

        label : int
        Label from the parent or child where name must be specified.

        If label is specified without name, this method will throw an error.

        get_ids : {array-like} of shape (n_samples,)
        Give specific index IDs to be extracted from the input.

        If no name or label is specified, it will return the IDs for all clusters.

        row_ids : {array-like} of shape (n_samples,)
        Specify the row_ids to be extracted from the square matrix of pairwise
        distances.

        This will override name, label and ID specification.

        col_ids : {array-like} of shape (n_samples,)
        Specify the col_ids to extracted from the square matrix of pairwise
        distances.

        This will override name, label and ID specification, but not row_ids.

        If row_ids are specified, the specified columns will be extracted only
        for those rows.

        Returns
        -------
        ndarray, ndarray
        Returns an array of the input text and their respective IDs
        
        """
        dst_out = self.distances
        name_array = None
        select_ids = np.arange(0, self.distances.shape[0])
        is_child = False
        idx = None
        row_col = False

        if name is not None:
            try:
                idx = self.parent_names.index(name)
            except:
                try:
                    idx = self.child_names.index(name)
                    is_child = True
                except:
                    raise ValueError('Name not found in parent and child.')
            
            if is_child:
                name_array = self.child_labels[:, idx]
            else:
                name_array = self.parent_labels[:, idx]
            name_mask = (name_array >= 0)
            dst_out = self.distances[:, name_mask]
            dst_out = dst_out[name_mask, :]
            name_array = name_array[name_mask]
            select_ids = select_ids[name_mask]

        if label is not None:
            if idx is None:
                raise ValueError('When label is specified, specific parent or child name must be provided to retrieve the values.')
            else:
                label_mask = (parent_labels == label)
                if np.any(label_mask):
                    dst_out = dst_out[:, label_mask]
                    dst_out = dst_out[label_mask, :]
                    select_ids = select_ids[label_mask]
                else:
                    logger.info('Label not found.')
                    dst_out = dst_out[:, label_mask]
                    select_ids = select_ids[label_mask]

        if get_ids is not None:
            id_mask = np.isin(select_ids, get_ids)
            select_ids = select_ids[id_mask]
            dst_out = dst_out[id_mask, :]
            dst_out = dst_out[:, id_mask]

        if row_ids is not None:
            row_col = True
            dst_out = self.distances[row_ids, :]
            select_ids = np.tile(np.arange(0, self.distances.shape[0]), self.distances.shape[0]).reshape(self.distances.shape)  
            select_ids = select_ids[row_ids, :]

        if col_ids is not None:
            if row_col:
                dst_out = dst_out[:, col_ids]
                select_ids = select_ids[:, col_ids]
            else:
                dst_out = self.distances[:, col_ids]
                select_ids = np.tile(np.arange(0, self.distances.shape[0]), self.distances.shape[0]).reshape(self.distances.shape)
                select_ids = select_ids[:, col_ids]
            

        return dst_out, select_ids
            
            
    def get_embeddings(self, name=None, label=None, get_ids=None):
        """
        Get the text embeddings by parent or child name, label and index ID.

        If neither is specified, all input is returned.

        Parameters
        ----------
        name : str
        Name of parent or child from parent_names or child_names.

        label : int
        Label from the parent or child where name must be specified.

        If label is specified without name, this method will throw an error.

        get_ids : {array-like} of shape (n_samples,)
        Give specific index IDs to be extracted from the input.

        If no name or label is specified, it will return the IDs for all clusters.

        Returns
        -------
        ndarray, ndarray
        Returns an array of the text embeddings and their respective IDs
        
        """
        embed_out = self.embeddings
        select_ids = np.arange(0, embed_out.shape[0])
        idx = None
        is_child = False
        name_array = np.array([])

        if name is not None:
            try:
                idx = self.parent_names.index(name)
            except:
                try:
                    idx = self.child_names.index(name)
                    is_child = True
                except:
                    raise ValueError('Name not found in parent and child.')
            if is_child:
                name_array = self.child_labels[:, idx]
            else:
                name_array = self.parent_labels[:, idx]

            name_mask = (name_array >= 0)
            embed_out = embed_out[name_mask, :]
            select_ids = select_ids[name_mask]
            name_array = name_array[name_mask]

        if label is not None:
            if idx is None:
                raise ValueError('When label is specified, either a parent or a child name must be specified.')
            label_mask = (name_array == label)
            embed_out = embed_out[label_mask, :]
            select_ids = select_ids[label_mask]

        if get_ids is not None:
            id_mask = np.isin(select_ids, get_ids)
            embed_out = embed_out[id_mask, :]
            select_ids = select_ids[id_mask]


        return embed_out, select_ids

    def get_reduce_embeddings(self, name=None, label=None, get_ids=None):
        """
        Get the dimension reduction embeddings by parent or child name, label and index ID.

        If neither is specified, all input is returned.

        Parameters
        ----------
        name : str
        Name of parent or child from parent_names or child_names.

        label : int
        Label from the parent or child where name must be specified.

        If label is specified without name, this method will throw an error.

        get_ids : {array-like} of shape (n_samples,)
        Give specific index IDs to be extracted from the input.

        If no name or label is specified, it will return the IDs for all clusters.

        Returns
        -------
        ndarray, ndarray
        Returns an array of the dimension reduction embeddings and their respective IDs
        
        """
        embed_out = self.reduce_embeddings
        select_ids = np.arange(0, embed_out.shape[1])
        idx = None
        is_child = False
        name_array = np.array([])
        embed_array = embed_out

        if name is not None:
            try:
                idx = self.parent_names.index(name)
            except:
                try:
                    idx = self.child_names.index(name)
                    is_child = True
                except:
                    raise ValueError('Name not found in parent and child.')
            if is_child:
                name_array = self.child_labels[:, idx]
                embed_array = self.reduce_embeddings[idx+1]
            else:
                name_array = self.parent_labels[:, idx]
                embed_array = self.reduce_embeddings[idx]

            name_mask = (name_array >= 0)
            
            embed_out = embed_array[name_mask, :]
            select_ids = select_ids[name_mask]
            name_array = name_array[name_mask]

        if label is not None:
            if idx is None:
                raise ValueError('When label is specified, either a parent or a child name must be specified.')
            label_mask = (name_array == label)
            embed_out = embed_out[label_mask, :]
            select_ids = select_ids[label_mask]

        if get_ids is not None:
            id_mask = np.isin(select_ids, get_ids)

            if idx:
                embed_out = embed_out[id_mask, :]
            else:
                logger.warning('No parent or child was specified, this returns the reduction embeddings at the specified IDs for all iterations '+\
                            'with dimensions (iteration, index, xy). Some values may be Nan.')
                embed_out = embed_out[:, id_mask, :]
                
            select_ids = select_ids[id_mask]
    
        return embed_out, select_ids

    def get_pandas_data(self):
        """
        Get all data as a Pandas dataframe.

        Returns
        -------
        DataFrame
        A Pandas dataframe object.
        """
        return self.pandas_data
