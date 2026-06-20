from utilities import validate_metric, validate_input
from clusterfeatures import ClusterFeatures
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances



class ClusterVectorOps:
    """
    This performs some basic vector operations that are useful for interpreting EHDBSCAN outputs.

    They are just raw vector operations. However it can be combined with LabelGeneratorConfig to
    generate labels for drawing the tree of clusters.

    Parameters
    ----------
    cluster_features : ClusterFeatures
    Input instance of ClusterFeatures that has been fit.

    metric : str
    The global metric for all operations. Metric may also be defined for individual operations.

    Metric is expected to be the same expected by sci-kit DitanceMetric

    use_embedding : bool
    Whether to use the input text embeddings stored in the ClusterFeatures. Otherwise it
    will use the pairwise distances stored in the ClusterFeatures.

    Generally, using embeddings is expected to yield better results.

    Returns
    -------
    self : object
    Returns self.
    
    """
    def __init__(self,
                cluster_features: ClusterFeatures,
                metric: str = 'cosine',
                use_embedding: bool = False):
        self.cf = cluster_features
        self.metric = validate_metric(metric)        
        self.use_embedding = use_embedding

    def cluster_centroid_neighbors(self, top_n: int = 3,
                                   name: str = None,
                                   label: int = None,
                                   metric: str = None):
        """
        This will generate the neighbors for a parent or child cluster centroid, where the centroid
        is taken as the median.

        Parameters
        ----------
        top_n : int
        Top N neighbors to return

        name : str
        Parent or child name as stored in the ClusterFeatures instance.

        label : int
        Parent or child label from the available labels in the ClusterFeatures instance.

        metric : str
        The metric to be used for centroid neighbors. It will override the global metric specification.

        Expects the same input as accepted by sci-kit DistanceMetric.

        Returns
        -------
        IDs of centroid neighbors {array-like} of shape (n_samples,)
        These IDs are aligned with the input to the EHDBSCAN and stored in the ClusterFeatures instance.
        """
        use_metric = self.metric
        if metric:
            use_metric = metric

        use_metric = validate_metric(use_metric)
            
        X, ids = self.cf.get_distances(name=name, label=label)
        if self.use_embedding:
            X, ids = self.cf.get_embeddings(name=name, label=label)
            if len(X) <= 0:
                raise ValueError('The ClusterFeatures instance provided has no stored embeddings.')
        
        X_median = np.median(X, axis=0)
        X_dst = X_median
        if self.use_embedding:
            X_dst = pairwise_distances(X_median.reshape(1, -1), X, metric=use_metric)

        X_dst = X_dst.ravel()
        mask = (X_dst != 0)
        ids = ids[mask]
        X_dst = X_dst[mask]
        
        args = np.argsort(X_dst)[:top_n]
        
        if top_n < 0:
            args = np.argsort(X_dst)[top_n:]
        
        ids = ids[args]
        return ids

    
    def parent_child_transition(self, *, parent: tuple, child: tuple, top_n: int = 3, metric: str = None):
        """
        In EHDBSCAN, parent-child links are based on core distance-type metric that relies on cluster centroids.
        Therefore, it is valuable to find out what was the common topical link between the two clusters. It helps
        interpreting the cluster tree.

        Similarly it is also helpful to see, which topics are unique to the parent and the child. In experiments,
        it has been observed that topics unique to the child can be seen as the topical drift subtracted from the
        parent (parent_median - child_median) and vice versa.

        For instance, if the parent cluster's central theme is Policy and Law and child cluster's theme is
        Malicious practices in the Law profession, the common link between the two is Law & Legal. The topical drift is
        Ethics, this is the difference unique to the child and will be reflected in the child IDs returned by this
        method.

        This method performs these operations to return IDs unique to the parent and child and IDs common
        among the two.

        It works with embeddings and distances, but it is highly recommended to use embeddings. The only distance
        estimations are simplification and not entirely representative. A lot of accuracy is lost in trying to work
        back differences in absolute positions with relative position metrics like distances.

        Parameters
        ----------
        parent : tuple
        A tuple of parent name and one of its cluster label e.g. ('parent_0', 2)

        child : tuple
        A tuple of child name and one of its cluster label e.g. ('child_0', 3)

        top_n : int
        Top N IDs of differences and commonalities to return.

        metric : str
        The metric to be used for distance estimations. It will override the global metric.

        Expects the same input as accepted by sci-kit DistanceMetric.

        Returns
        -------
        IDs for Parent Difference, Child Difference, Parent-Child Commmon {array-like} of shape (n_samples,)
        IDs are returned in the sequence of the input stored in the ClusterFeats instance.
        
        """
        use_metric = self.metric
        if metric:
            use_metric = metric

        use_metric = validate_metric(use_metric)
            
        if not isinstance(parent, tuple):
            raise ValueError('Expects a tuple of size 2 with name at first index and label at second.')
        
        if not isinstance(child, tuple):
            raise ValueError('Expects a tuple of size 2 with name at first index and label at second.')

        p_ids = []
        c_ids = []
        common_ids = []

        if self.use_embedding:
            X_chk, _ = self.cf.get_embeddings()
            if len(X_chk) <= 0:
                raise ValueError('The ClusterFeatures instance provided has no stored embeddings.')
            X_parent, p_ids = self.cf.get_embeddings(name=parent[0], label=parent[1])
            X_child, c_ids = self.cf.get_embeddings(name=child[0], label=child[1])
            all_embed = np.concatenate((X_parent, X_child), axis=0)
            all_ids = np.concatenate((p_ids, c_ids), axis=0)

            X_p_median = np.median(X_parent, axis=0)
            X_c_median = np.median(X_child, axis=0)
            X_c_diff = X_p_median - X_c_median
            X_p_diff = X_c_median - X_p_median
            X_common = X_p_diff + X_c_diff

            X_p_dst = pairwise_distances(X_p_diff.reshape(1, -1), X_parent, metric=use_metric)
            X_p_dst = np.where(X_p_dst != 0., X_p_dst, np.inf)
            X_c_dst = pairwise_distances(X_c_diff.reshape(1, -1), X_child, metric=use_metric)
            X_c_dst = np.where(X_c_dst != 0., X_c_dst, np.inf)
            X_cm_dst = pairwise_distances(X_common.reshape(1, -1), all_embed, metric=use_metric)
            X_cm_dst = np.where(X_cm_dst != 0., X_cm_dst, np.inf)

            p_args = np.argsort(X_p_dst.ravel())[:top_n]
            c_args = np.argsort(X_c_dst.ravel())[:top_n]
            cm_args = np.argsort(X_cm_dst.ravel())[:top_n]

            p_ids = p_ids[p_args]
            c_ids = c_ids[c_args]
            common_ids = all_ids[cm_args]
        else:
            X_parent, p_ids = self.cf.get_distances(name=parent[0], label=parent[1])
            X_child, c_ids = self.cf.get_distances(name=child[0], label=child[1])

            X_all, all_ids = self.cf.get_distances(row_ids=p_ids, col_ids=c_ids)
            X_dst = np.where(X_all != 0., X_all, np.inf)
            X_cm_args = np.argsort(X_dst.ravel())[:top_n]
            common_ids = all_ids.ravel()[X_cm_args]

            p_args = np.argsort(np.median(X_parent, axis=0))
            p_ids = p_ids[p_args][:top_n]
            
            c_args = np.argsort(np.median(X_child, axis=0))
            c_ids = c_ids[c_args][:top_n]
            

        return p_ids, c_ids, common_ids
