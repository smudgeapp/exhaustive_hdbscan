import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances
from typing import List, Optional
from typing import Any
import logging
from copy import deepcopy
from utilities import validate_input, validate_metric, Encoder, Reducer
from clusterfeatures import ClusterFeatures
from labelgenerator import LabelGeneratorConfig

try:
    if logger:
        logger.handlers.clear()
        del logger
except:
    pass
    
logger = logging.getLogger(__name__)
streamer = logging.StreamHandler()
logger.addHandler(streamer)
logger.propagate = False



class EHDBSCAN:

    """
    Exhaustive HDBSCAN performs iterative clustering. The noise points left over from each iteration
    are remapped to a new density structure. At each step the clusters from the current iteration are
    assigned parents from the previous iteration. This results in a cluster of linked clusters.

    This implementation uses HDBSCAN from the sci-kit library.

    Overall, this lib relies heavily on sci-kit.
    """
    
    def __init__(self, *,
                 metric: str = 'cosine',
                 tree_metric: str = 'cosine',
                 min_clustersize_multiplier: float = 0.05,
                 max_clustersize_multiplier: float = 5.0,
                 min_samples: int = 5,
                 n_jobs: int = None,
                 log_level: int = logging.DEBUG,
                 cluster_core_k: int = -1,
                 reducer: Reducer = None,
                 encoder: Encoder = None
                 ):

        """
        Initialize the EHDBSCAN to perform iterative clustering.

        Parameters
        ----------

        metric : str
        The metric to be used by HDBSCAN. It must be one acceptable by sci-kit pairwise_distances method.

        If precomputed is passed, the fit method will accept a square matrix of pairwise distances.

        tree_metric : str
        The metric to be used for forming parent-child links at each clustering step.

        It must be one acceptable by pairwise_distances method.

        If precomputed, this will not apply.

        min_clustersize_multiplier : float
        The mutliplier applied to the full length of the input at each clustering step.

        max_clustersize_multiplier : float
        The multiplier applied to the minimum cluster size at each clustering step.

        min_samples : int
        The starting number of minimum samples to be considered by HDBSCAN as neighbors.

        This will automatically be reduced by 1 to a minimum of 2 through each clustering step.

        n_jobs : int
        Number of threads. Application is same as sci-kit.

        log_level : int
        Set the logging level. Same values apply as default Python logging, DEBUG, INFO, WARNING, ERROR

        cluster_core_k : int
        This identifies nearest neighbors to be considered for forming parent-child links.

        The core distance between clusters.

        reducer : Reducer
        The dimension reduction object. The Reducer must be subclassed with desired reduction method.

        See examples for implementation.

        While dimension reduction is not required for HDBSCAN, it is highly recommended. Typically,
        UMAP is very effective with HDBSCAN. But UMAP implementations may introduce non-determinism.
        
        encoder : Encoder
        The text embedder object. The Encoder must be subclassed with desired encoding method/model.

        If no encoder is used, text embeddings or pairwise distances must be provided.

        See examples for implementation.

        Returns
        -------

        self : object
        Returns self.

        
        """
        self.metric = validate_metric(metric)
        self.tree_metric = validate_tree_metric(tree_metric)
        self.precomputed = True if self.metric == 'precomputed' else False
        self.min_clustersize_multiplier = min_clustersize_multiplier
        self.max_clustersize_multiplier = max_clustersize_multiplier
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        logger.setLevel(log_level)
        streamer.setLevel(log_level)
        self.data = pd.DataFrame()
        self.parent_labels = 0
        self.child_labels = 0
        self.parent_names = []
        self.child_names = []
        self.cluster_core_k = cluster_core_k
        self.reducer = reducer
        self.encoder = encoder
        self.models = {}
        self.isfit = False

    
    def fit(self, X, y=None):
        """
        The fit method, performs the iterative clustering.

        Parameters
        ----------

        X : {array-like} of shape (n_samples, n_features) or ndarray of shape (n_samples, n_samples)
        A feature array of text, embeddings or pairwise distances.

        y : None
        Ignored

        Returns
        -------

        self : object
        Returns self.
        
        """
        
        X_recur, is_embedding = validate_input(X)
        idx = np.arange(0, len(X_recur))
        mask = np.ones(idx.shape, dtype=bool)
        itr_idx = idx[mask]
        
        self.cluster_feats = ClusterFeatures()

        if self.precomputed:
            self.cluster_feats._add_distances(X_recur)
        else:
            if is_embedding == False:
                self.cluster_feats._add_input(X)
                if self.encoder == None:
                    raise ValueError('Expects Encoder to be specified when text is input. OR directly input embeddings.')
                X_recur = self.encoder.encode(X_recur)
                self.cluster_feats._add_embeddings(X_recur)
                
            pdist = pairwise_distances(X_recur, metric=self.tree_metric)
            self.cluster_feats._add_distances(pdist)
        
        recur = True
        itr = 0        
        min_samps = 5
        min_cluster_size_m = 0.05
        max_cluster_size_m = 5
        min_size = int(round(len(X_recur) * min_cluster_size_m))
        
        if self.min_clustersize_multiplier:
            min_cluster_size_m = self.min_clustersize_multiplier
            min_size = int(round(len(X_recur) * min_cluster_size_m))

        if self.max_clustersize_multiplier:
            max_cluster_size_m = self.max_clustersize_multiplier

        if self.min_samples:
            min_samps = self.min_samples
        
        
        while recur:
            model = {'hdb': None, 'reducer': None}
            
            if self.reducer:
                redr = self.reducer.fit(X_recur)
                X_hdb = self.reducer.transform(X_recur)
                redr_fill_shape = (idx.shape[0], X_hdb.shape[-1])
                if len(X_hdb.shape) == 1:
                    redr_fill_shape = (idx.shape[0],)
                redr_fill = np.full(shape=redr_fill_shape, fill_value=np.nan)
                redr_fill[itr_idx] = X_hdb
                self.cluster_feats._add_reduce_embeddings(redr_fill)
                model['reducer'] = deepcopy(self.reducer)

            hdb = HDBSCAN(metric=self.metric,
                                min_cluster_size=min_size,
                                min_samples=min(min_samps, min_size // 2),
                                n_jobs=self.n_jobs,
                                max_cluster_size=int(min_size * max_cluster_size_m))
            
            hdb.fit(X_hdb)
            model['hdb'] = deepcopy(hdb)

            labels = hdb.labels_
            
            if max(labels) < 0:
                recur = False
                logger.info(f'Exiting clustering loop. No more clusters found.')
                break

            current_labels = list(np.unique(labels))
            
            if -1 in current_labels:
                current_labels.remove(-1)
            else:
                logger.info(f'FINAL ITERATION {itr}. All values consumed.')
                recur = False
            
            fill_vec = np.full(shape=idx.shape, fill_value=-2)
            fill_vec[itr_idx] = labels
            self.cluster_feats._add_parent(fill_vec)
            self.models[self.cluster_feats.parent_names[-1]] = model
            
            logger.info(f'Iteration: {itr}, Clusters: {current_labels}, Noise: {np.sum((labels == -1)).item()}')
            
            if itr > 0:
                child_vec = np.full(shape=idx.shape, fill_value=-2)
                
                for label in current_labels:
                    this_child_mask = (fill_vec == label)
                    this_child_dist = []
                    for plabel in parent_labels:                    
                        this_parent_mask = (parent_clusters == plabel)
                        this_child_dist.append(self._cluster_sim_score_dst(self.cluster_feats.get_distances()[0], parent=idx[this_parent_mask],
                                                                           child=idx[this_child_mask]))                    
                    max_idx = np.argmax(this_child_dist)
                    logger.info(f'For child_{itr-1}={label}, parent_{itr-1}={max_idx}')
                    child_vec[this_child_mask] = parent_labels[max_idx]
                
                self.cluster_feats._add_child(child_vec)
                
            parent_labels = current_labels
            parent_clusters = fill_vec
            
            mask = (labels == -1)
            itr_idx = itr_idx[mask]
            
            X_recur = X_recur[mask, :]
            
            if self.precomputed:
                X_recur = X_recur[:, mask]
            
            if len(X_recur) <= 5:
                recur = False
                logger.info(f'Exiting clustering loop. Minimum values reached. Remaining length = {len(X_recur)}')
                break

            min_size = int(round(len(X_recur) * min_cluster_size_m))
            if min_size < 5:
                min_size = 5

            max_size = min_size * 5
            if max_size > len(X_recur):
                max_size = len(X_recur)

            min_samps -= 1
            if min_samps <= 2:
                min_samps = 2

            itr += 1

        
        self.isfit = True

        return self
    
    def approx_predict(self, X):
        """
        Approximate prediction will assign unseen data to existing clusters.

        HDBSCAN is not built for predictions, so this is not a perfect transform,
        only an approximate prediction.

        It expects data to be passed in the same format as the fit method. But if text was
        passed to fit, it will still work with embeddings.

        This method will not work with pairwise distances where metric is precomputed.

        If EHDBSCAN fit was executed with pairwise distances, this method will not work,
        even if text or embeddings are passed for prediction.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
        An array of text or embeddings.

        Returns
        -------
        ClusterFeatures : object
        An instance of ClusterFeatures for the prediction X.
        
        """
        
        if self.isfit == False:
            raise TypeError('Cannot be called on un-fitted model. Please call fit first.')
            
        if self.precomputed:
            raise ValueError('This method is not compatible with precomputed distances. It requires raw features to calculate pair distances.')
        
        cluster_feats = ClusterFeatures()

        X_input, is_embedding = validate_input(X)
        
        if is_embedding == False:
            cluster_feats._add_input(X_input)
            X_input = self.encoder.encode(X_input)
            cluster_feats._add_embeddings(X_input)
            pair_dist = pairwise_distances(X_input, metric=self.tree_metric)
            cluster_feats._add_distances(pair_dist)
        
        X_recur = deepcopy(X_input)
        
        idx = np.arange(0, X_input.shape[0])
        itr_idx = np.arange(0, X_input.shape[0])
        
        min_samps = self.min_samples

        id_mask = 0
        
        parent_labels = []
        
        for i, ex_parent in enumerate(self.cluster_feats.parent_names):
            if len(X_recur) <= 5:
                logging.info('Exiting clustering loop. Minimum values reached.')
                break

            ex_embeddings = ''
            ex_ids = ''
            y_dst = ''
            y_nbrs = NearestNeighbors(metric=self.metric, n_neighbors=min_samps)
            ex_nbrs = NearestNeighbors(metric=self.metric, n_neighbors=min_samps)
            exy_dst = ''
            
            if self.reducer:
                ex_embeddings, ex_ids = self.cluster_feats.get_reduce_embeddings(name=ex_parent)
                y_redr = self.models[ex_parent]['reducer'].transform(X_recur)
                y_redr_shape = (idx.shape[0], y_redr.shape[-1])
                if len(y_redr.shape) == 1:
                    y_redr_shape = (idx.shape[0],)
                emb_fill = np.full(shape=y_redr_shape, fill_value=np.nan)
                emb_fill[itr_idx] = y_redr
                cluster_feats._add_reduce_embeddings(emb_fill)
                y_nbrs.fit(y_redr)
                y_dst, _ = y_nbrs.kneighbors(y_redr)
                exy_dst = pairwise_distances(y_redr, ex_embeddings, metric=self.metric)
            else:
                ex_embeddings, ex_ids = self.cluster_feats.get_embeddings(name=ex_parent)
                y_nbrs.fit(X_recur)
                y_dst, _ = y_nbrs.kneighbors(X_recur)
                exy_dst = pairwise_distances(X_recur, ex_embeddings, metric=self.metric)
                
            ex_labels, _ = self.cluster_feats.get_parent(name=ex_parent)
            ex_nbrs.fit(ex_embeddings)
            ex_dst, _ = ex_nbrs.kneighbors(ex_embeddings)

            this_y_dist = y_dst[:, -1]
            this_y_dist = this_y_dist.reshape(-1, 1)
            
            this_ex_dist = ex_dst[:, -1]
            
            ext_p_dist = np.tile(this_ex_dist, this_y_dist.shape[0])
            ext_p_dist = ext_p_dist.reshape(-1, this_ex_dist.shape[0])
            
            mrd = np.maximum(np.maximum(this_y_dist, ext_p_dist), exy_dst)
            
            min_mrd = np.argmin(mrd, axis=1)
            
            ex_core = this_ex_dist[min_mrd]
            this_ex_labels = ex_labels[min_mrd]

            labels = np.where(this_y_dist.ravel() < ex_core, this_ex_labels, -1)

            child_labels = list(np.unique(labels))

            logger.info(f'Iteration {i}, Clusters: {child_labels}')

            if max(child_labels) < 0:
                logger.info('Exiting clustering loop. No more clusters found.')
                break

            if -1 in child_labels:
                child_labels.remove(-1)
            else:
                logging.info(f'FINAL ITERATION {i}. All values consumed.')

            child_labels = [lb for lb in child_labels if lb >= 0]

            fill_vec = np.full(shape=idx.shape[0], fill_value=-2)
            fill_vec[itr_idx] = labels
            
            cluster_feats._add_parent(fill_vec)

            if i > 0:
                child_vec = np.full(shape=idx.shape[0], fill_value=-2)
                all_dist = cluster_feats.get_distances()[0]
                for label in child_labels:
                    this_child_mask = (fill_vec == label)
                    this_child_dst = []
                    for plabel in parent_labels:
                        _, this_parent_ids = cluster_feats.get_parent(name=cluster_feats.parent_names[i-1], label=plabel)
                        this_child_dst.append(self._cluster_sim_score_dst(all_dist, parent=this_parent_ids, child=idx[this_child_mask]))
                    id_max = np.argmax(this_child_dst)
                    child_ids = idx[this_child_mask]
                    child_vec[child_ids] = parent_labels[id_max]
                    logging.info(f'Child = {label}, Parent = {id_max}')
                cluster_feats._add_child(child_vec)
                
            parent_labels = child_labels
            min_samps -= 1
            id_mask = (labels < 0)
            itr_idx = itr_idx[id_mask]
            X_recur = X_recur[id_mask, :]

        return cluster_feats
                    
                    
    def _cluster_sim_score_dst(self, full_dst, *, parent, child):
        child_parent_codist = full_dst[np.ix_(child, parent)]
        
        child_parent_mindist = np.amin(child_parent_codist, axis=1)

        parent_core = full_dst[parent, parent]
        child_core = full_dst[child, child]

        parent_core = np.sort(parent_core, axis=1)
        child_core = np.sort(child_core, axis=1)

        if self.cluster_core_k > 0:
            if self.cluster_core_k > full_dst.shape[1]:
                raise ValueError('Cluster core distances k must be less than the total input length.'+\
                                 'Typically a value of 5 is a good balance between reach and density.')
            parent_core = parent_core[:, self.cluster_core_k+1]
            child_core = child_core[:, self.cluster_core_k+1]

        parent_core_median = np.tile(np.array(np.median(parent_core)), child_parent_mindist.shape[0])
        child_core_median = np.tile(np.array(np.median(child_core)), child_parent_mindist.shape[0])

        mutual_r_dist = np.max(np.vstack((child_parent_mindist, child_core_median, parent_core_median)), axis=0)
        density = 1 / mutual_r_dist
        return np.median(density)

    
    def _annotate_box(self, ax, text, cds, dim=200, fontsize=10, box_style="round,pad=0.3", box_color='aliceblue', box_border_color='black', box_border_width=1):

        ann = ax.annotate(text, xy=cds,
                           bbox=dict(boxstyle="round,pad=0.3", fc=box_color, ec=box_border_color, lw=box_border_width),
                           fontsize=fontsize,
                           ha='center', va='center',
                          zorder=5,
                            wrap=True)

        ann._get_wrap_line_width = lambda: dim
        return ann 
   

  
    def plot_tree(self,
                  cluster_features: ClusterFeatures = None,
                  labeler: LabelGeneratorConfig = None,
                 figsize: tuple = (20, 10),
                  horizontal_spacing: int = 3,
                  vertical_spacing: int = 1,
                  ax : matplotlib.pyplot.axis = None,
                  label_font_size: int = 10,
                  label_box_dim: int = 100,
                  label_box_style: str = "round,pad=0.3",
                  label_box_color: str = 'aliceblue',
                  label_box_border_color: str = 'black',
                  label_box_border_width: int = 1
                 ):

        """
        It plots the tree of clusters with parent-child links.

        Parameters
        ----------
        cluster_features : ClusterFeatures
        If ClusterFeatures are provided, it will plot the tree of the input. If
        not provided, it will plot the tree of the cluster features from the fit.

        labeler : LabelGeneratorConfig
        If labeler is provided, they will be displayed on the tree nodes.

        Labeler must be a subclass of LabelGeneratorConfig.

        See examples for implementation

        figsize : tuple
        A tuple setting the figure size. Same as matplotlib figsize.

        horizontal_spacing : int
        Set the horizontal spacing between tree nodes.

        vertical_spacing : int
        Set the vertical spacing between tree levels.

        ax : matplotlib.axes.Axes
        Set the axis object on which to draw the tree.

        label_font_size : int
        Font size for node labels.

        label_box_dim : int
        Box size for node boxes.

        label_box_style : str
        Box style for label boxes. Matplotlib compatible.

        label_box_color : str
        Box inner color. Matplotlib compatible.

        label_box_border_color : str
        Box border line color. Matplotlib compatible.

        label_box_border_width : int
        Box border line width.

        

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes
        The instance of figure and axis on which the tree is drawn.

        
        """
        
        cluster_feats = self.cluster_feats
        
        if cluster_features:
            cluster_feats = cluster_features
        
        parent_count = 0
        child_count = 0
                
        vert_spacing = vertical_spacing
        hori_spacing = horizontal_spacing
        
        annotate_labels = []
        
        if labeler:
            if not isinstance(labeler, LabelGeneratorConfig):
                raise ValueError('Expects labeler to be of type LabelGeneratorConfig.')
            annotate_labels = labeler.compute(cluster_feats)
        
        ann_x = 0
        ann_y = parent_count
        
        x_range = [0, 0]
        annotations = []
        
        
        fig, axs = plt.subplots(figsize=figsize)
        if ax:
            axs = ax
        parent_boxes = {}
        total_itr = 0
        
        for i, parent in enumerate(cluster_feats.parent_names):
            parent_all, _ = cluster_feats.get_parent(name=parent)
            parent_all = parent_all[(parent_all >= 0)]
            parent_labels = list(np.unique(parent_all))
            this_parent = {}
            for k, label in enumerate(parent_labels):
                parent_text = ''
                if ann_x > 0:
                    ann_x += k * -hori_spacing
                else:
                    ann_x += k * hori_spacing
                ann_label = k
                ann_label = f'Parent{i}_Cluster{k}'
                if labeler:
                    ann_label += f'\n{annotate_labels[total_itr]}'
                    
                self._annotate_box(axs, ann_label, (ann_x, ann_y),
                                   box_style=label_box_style,
                                   box_color=label_box_color,
                                   box_border_color=label_box_border_color,
                                   box_border_width=label_box_border_width,
                                   dim=label_box_dim,
                                   fontsize=label_font_size)

                this_parent[label] = {'points': [ann_x, ann_y], 'label': ann_label}
                total_itr += 1

            parent_boxes[i] = this_parent
            ann_y -= vert_spacing
            hori_spacing += 1
            ann_x = 0

        for parent in parent_boxes.keys():
            if parent < len(parent_boxes.keys()) - 1:
                box_locs = parent_boxes[parent]
                parent_labels, _ = cluster_feats.get_parent(name=f'parent_{parent}')
                child_labels, child_pts = cluster_feats.get_child(name=f'child_{parent}')
                    
                for label in box_locs.keys():
                    parent_loc = box_locs[label]['points']                    
                    child_cluster_mask = (child_labels == label)
                    sub_parent, sub_parent_pts = cluster_feats.get_parent(name=f'parent_{parent+1}')
                    csp_mask = np.isin(sub_parent_pts, child_pts[child_cluster_mask])
                    sub_parent = sub_parent[csp_mask]
                    sub_parents = np.unique(sub_parent)
                    sub_parent_locs = parent_boxes[parent+1]
                    for sub in sub_parents:                       
                        child_loc = sub_parent_locs[sub]['points']
                        axs.plot([parent_loc[0], child_loc[0]], [parent_loc[1], child_loc[1]], 'k-', lw=1)
        plt.axis('off')
        plt.show()
        
        return fig, axs
        
