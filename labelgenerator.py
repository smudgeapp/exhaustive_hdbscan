import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp
from clusterfeatures import ClusterFeatures
from utilities import validate_input, Encoder, Reducer
from typing import List, Optional
from typing import Any




class LabelGeneratorConfig:
    """
    The base class for generating labels compatible with EHDBSCAN.

    """
    def __init__(self):
        pass
        
    def compute(self, cluster_features: ClusterFeatures):

        """
        All subclasses must implement this method.
        
        It expects an instance of ClusterFeatures and returns a list of
        labels in the same sequence as the clustering operation. 

        Cluster Iteration > Each Cluster in ascending sequence [0, 1, 2...].

        Returned list must be single level, not nested. This is must for
        compatibility with EHDBSCAN.

        See examples for implementation.

        Parameters
        ----------
        cluster_features : ClusterFeatures
        Instance of ClusterFeatures. This is must for compatibility with EHDBSCAN.

        Returns
        -------
        output : list
        Returns a single level list of generated labels for cluster visualization or other
        downstream tasks. Labels must be in the sequence of the clustering procedure.
        """
        

        raise NotImplementedError('Subclass must implement compute method.'+\
                                  'It must take ClusterFeatures as input and return a list of labels in the sequence cluster iteration, cluster label...')


class MMR1(LabelGeneratorConfig):
    """
    Maximal Marginal Relevance implementation subclassed to LabelGeneratorConfig. It filters the
    labels for relevance and diversity.

    This is a generalized standalone implementation and will work without EHDBSCAN inputs. But
    the compute method will only work with an instance of ClusterFeatures. Fit and transform may still
    be called with raw inputs.

    Parameters
    ----------
    lda : float
    lambda value for MMR. Ranges from 0 to 1.

    0 = Only diversity; 1 = Only relevance.

    encoder : Encoder {Optional}
    Expects a subclass of Encoder.

    If no encoder is passed, it expects embeddings as input. In this case,
    labels are index of the passed embeddings.

    metric : str
    Metric to be used for similarity. Expects metric acceptable by sci-kit
    DistanceMetric.

    top_n : int
    Top N similarity labels to be output.


    Attributes
    ----------
    label_sim : {array-like} of shape (n_samples, n_samples)
    Pairwise similarity matrix of input.

    label_embed : {array-like} of shape (n_samples, n_features)
    Label embeddings. Only available if Encoder is passed.

    
    """
    
    def __init__(self, *,
                 lda: float = 0.5,
                 encoder: Encoder = None,
                 metric: str = 'cosine',
                top_n: int = 3):
        self.lda = lda
        self.encoder = encoder
        self.top_n = top_n
        self.metric = metric
        self.metric_is_max = metric_is_max
        

    def fit(self, X):
        """
        Creates the similarity matrix from inputs. Expects text input or
        numeric embeddings.

        X : {array-like} of shape (n_samples,) for text OR (n_samples, n_features) for embeddings
        Expects text or numeric embeddings. If text is provided, an Encoder must also be specified.

        """
        X, is_embedding = validate_input(X)
        
        if is_embedding:
            self.labels = np.arange(0, X.shape[0])
            self.label_embed = X
        else:
            if self.encoder:
                self.labels = X
                self.label_embed = self.encoder.encode(X)
            else:
                raise ValueError('Must specify an Encoder when text input is passed.')
            
        self.label_sim = pairwise_distances(self.label_embed, metric=self.metric)


    def transform(self, y):
        #TODO transform based only on pairwise distances
        """
        Get the MMR labels.

        y : {array-like} of shape (n_samples,) for text OR (n_samples, n_features) for embeddings
        Expects text or numeric embeddings. If text is provided, an Encoder must also be specified.
        
        """
        label_idx = []
        y, is_embedding = validate_input(y)
        y_embed = y
        if is_embedding == False:
            if self.encoder:
                y_embed = self.encoder.encode(y)
            else:
                raise ValueError('Must specify an Encoder when text input is passed.')

        if len(y_embed.shape) < 2:
            y_embed = y_embed.reshape(1, -1)

        target_sim = pairwise_distances(y_embed, self.label_embed, metric=self.metric)
        idx = np.argmin(target_sim, axis=1)
        label_idx.append(idx[0])
        
        remainder = np.arange(0, self.label_embed.shape[0])
        remainder_mask = np.isin(remainder, label_idx, invert=True)
        remainder = remainder[remainder_mask]        
        
        for i in range(self.top_n):
            t_sim = target_sim[:, remainder]
            d_sim = self.label_sim[np.ix_(label_idx, remainder)]
            d_sim = np.min(d_sim, axis=0)
            mmr_scores = (self.lda * t_sim) - ((1 - self.lda) * d_sim)
            top_score = np.argsort(mmr_scores.ravel())[0]
            label_idx.append(remainder[top_score])
            remainder_mask = np.isin(remainder, label_idx, invert=True)
            remainder = remainder[remainder_mask]
        
        sorted_labels = self.labels[label_idx]
        return sorted_labels

    def _preprocess(self, cluster_feats: ClusterFeatures):
        has_embedding = False
        ot = []

        if len(cluster_feats.get_embeddings()) > 0:
            has_embedding = True
        
        for parent in cluster_feats.parent_names:
            p_labels, _ = cluster_feats.get_parent(name=parent)
            p_labels = p_labels[(p_labels >= 0)]
            u_labels = np.unique(p_labels)
            for label in u_labels:
                item = {}
                _, p_ids = cluster_feats.get_parent(name=parent, label=label)
                p_text, _ = cluster_feats.get_input(get_ids=p_ids)
                item['texts'] = list(p_text)
                item['class_text'] = ' '.join(item['texts'])

                if has_embedding:            
                    p_embeddings, _ = cluster_feats.get_embeddings(get_ids=p_ids)
                    item['embeddings'] = p_embeddings
                    
                item['parent'] = parent
                item['label'] = label
                ot.append(item)
                    
        return ot, has_embedding
       
           

    def compute(self, cluster_feats: ClusterFeatures):
        """

        Performs the fit and transforms the input labels as MMR labels and outputs
        list of labels compatible with EHDBSCAN. It expects a ClusterFeatures input.

        For ranking, the entire cluster text is clubbed and encoded to compare against
        individual items in the cluster. If an encoder is provided, the combined cluster
        text is encoded with this encoder. If an encoder is not provided, the average
        embeddings of individual cluster items is used.

        If the ClusterFeatures instance provided has no embeddings, an encoder must be
        provided.

        Parameters
        ----------
        cluster_feats : ClusterFeatures
        An instance of ClusterFeatures.

        Returns
        -------
        MMR Ranked Labels : list
        A single level list of MMR ranked labels for each cluster in sequence
        iteration > cluster.
        
        """

        ot, has_embedding = self._preprocess(cluster_feats)
        X = []
        y = []
        all_text = [text for x in ot for text in x['texts']]
        y = [x['class_text'] for x in ot]
        
        if has_embedding:
            X = np.concatenate([item['embeddings'] for item in ot], axis=0)
            self.fit(X)
            if self.encoder is None:
                y = [np.mean(x['embeddings'], axis=0) for x in ot]
        else:
            if self.encoder:
                self.fit(all_text)
            else:
                raise ValueError('Neither the ClusterFeatures had embeddings, nor an Encoder was provided with MMR. One of the two must be available.')
        labels = []
        for item in y:
            label = self.transform(np.array(item))
            labels.append(' '.join([all_text[x] for x in label]))
        
        return labels            


class ClassTfidf(LabelGeneratorConfig):
    """

    Class Term Frequency Inverse Document Frequency implementation subclassed to LabelGeneratorConfig. It
    assigns descriptive labels for each cluster.

    This is a generalized standalone implementation and will work without EHDBSCAN inputs. But
    the compute method will only work with an instance of ClusterFeatures. Fit and transform may still
    be called with raw inputs.

    It uses the sci-kit CountVectorizer and some of those parameters are exposed here.

    Parameters
    ----------
    top_n : int
    Top N labels to output.

    min_df : float
    Minimum frequency threshold to include terms. Same as sci-kit.

    max_df : float
    Maximum frequency threshold to include terms. Same as sci-kit.

    ngram_range : tuple
    Range of ngrams to include. Same as sci-kit.

    stop_words : str, list
    List of words to exclude. Same as sci-kit.

    token_pattern: str
    Regex pattern for including words. Same as sci-kit.

    lowercase : bool
    Converts input to lowercase. Same as sci-kit.

    max_features : int
    Max length of features to retain. Same as sci-kit.

    mmr : MMR
    An instance of MMR.

    This will apply on the class labels after the ClassTfidf procedure has run.

    This will only work with the compute method. Simple fit and transform will
    not execute an MMR ranking.

    Attributes
    ----------
    cW : scipy sparse matrix of shape (n_samples, n_features)
    The class weights generated after the fit method is called.

    self.features : {array-like} of shape (n_features,)
    The vectorized vocabulary. Only callable after fit method is called.

    
    """
    def __init__(self, *, 
                 top_n: int = 3, 
                 min_df: float = 0.20, 
                 max_df: float = 0.50, 
                 ngram_range: tuple = (1, 2),
                stop_words: Any = 'english',
                 token_pattern: str = r'(?u)\b[a-zA-Z]{4,}\b',
                 lowercase: bool = True,
                 max_features: int = None,
                 mmr: MMR = None):
        super().__init__()
        self.top_n = top_n
        self.mmr = mmr
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.isfit = False
        self.lowercase = lowercase
        self.max_features = max_features

    
    def _preprocess(self, cluster_feats: ClusterFeatures):
        if not isinstance(cluster_feats, ClusterFeatures):
            raise ValueError('Expected input of type ClusterFeatures. Raw values can only be passed to fit and transform.')
            
        class_texts = []
        
        for parent in cluster_feats.parent_names:
            labels, _ = cluster_feats.get_parent(name=parent)
            label_mask = (labels >= 0)
            labels = labels[label_mask]
            for label in np.unique(labels):
                item = {}
                item['parent'] = parent
                item['label'] = label
                raw_text, _ = cluster_feats.get_input(name=parent, label=label)
                c_text = ', '.join(raw_text)
                item['text'] = c_text.lower()
                class_texts.append(item)
        
        return class_texts
                
    def fit(self, X):
        """
        Fits a corpus of docs by vectorizing ngrams and generating normalized TF-IDF vectors
        for each doc.

        It also caches the fit vectorizer and the Class Weights matrix for transform.

        Unlike regular TF-IDF, the Class TF-IDF applies over a class of documents weighting the
        regular TF-IDF by a class-relative weighting formula.

        This is a suitable TF-IDF labelling scheme for clusters, where each cluster is treated as
        a class. Labels are then generated for each cluster relative to all other clusters.

        Parameters
        ----------
        X : {array-like} of shape (n_samples,)
        Only text input accepted.

        Returns
        ----------
        TF-IDF {array-like} of shape (n_samples, n_features)
        The array matrix of Class TF-IDF normalized scores of each document.
        
        """
        X, _ = validate_input(X, classtfidf=True)
        
        self.vectorizer = CountVectorizer(stop_words=self.stop_words, token_pattern=self.token_pattern,
                                     ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
                                          lowercase=self.lowercase,
                                          max_features=self.max_features)
        self.vectorizer.fit(X)
        X_transform = self.vectorizer.transform(X)
        self.features = self.vectorizer.get_feature_names_out()

        X_norm = normalize(X_transform, norm='l1', axis=1)
        A = np.mean(np.sum(X_transform, axis=1))
        f = np.sum(X_transform.toarray(), axis=0)
        self.cW = sp.diags(np.log(1 + A / f))
        cTf = X_norm @ self.cW
        self.isfit = True
        return cTf

    def transform(self, y):
        """
        Transform unseen data to current Class TF-IDF. It vectorizes the input to the current
        fit vocabulary and applies the Class TF-IDF weights to obtain a vector of TF-IDF for the
        current input.

        This method does not assign new data to classes by design. That is a downstream task and really upto task
        and user preferences. Class assignment may be executed on similarity between TF-IDF vectors, label
        distances or even label embedding vector operations. Class assignment may not require doing this transform
        at all.

        Parameters
        ----------
        y : {array-like} of shape (n_samples,)
        Input text array for transforming along the current Class TF-IDF structure.

        Returns
        -------
        TF-IDF {array-like} of shape (n_samples, n_features)
        
        """
        
        if self.isfit == False:
            raise ValueError('Must call fit first.')
        y, _ = validate_input(y, classtfidf=True)
        y_ct = self.vectorizer.transform(y)
        y_norm = normalize(y_ct, norm='l1', axis=1)
        yctf = y_norm @ self.cW
        
        return yctf
    

    def compute(self, cluster_feats: ClusterFeatures):
        """
        Extracts data from ClusterFeatures, packs them into classes by their cluster and generates
        Class TF-IDF labels.

        If MMR is passed, it will perform an MMR ranking after class labels have been generated.

        Parameters
        ----------
        cluster_feats : ClusterFeatures
        An instance of ClusterFeatures.

        Returns
        -------
        Class Labels : list
        A single level list of top_n class labels for each cluster of each iteration in sequence
        iteration > cluster. 
        """
        class_texts = self._preprocess(cluter_feats)
        texts = [x['text'] for x in class_texts]
        
        cTf = self.fit(texts)
        
        ctf_labels = []
        ot_labels = []
        
        for i, score in enumerate(cTf):
            score_arr = score.toarray().ravel()
            arr = np.argsort(score_arr)[::-1]
            labels = self.features[arr]
            score_arr = score_arr[arr]
            mask = (score_arr > 0)
            labels = labels[mask]
            ctf_labels.append(labels)
            
        if self.mmr:
            self.mmr.fit(self.features)
            for l in ctf_labels:
                ot_labels.append(' '.join(self.mmr.transform(l)))
        else:
            ot_labels = [' '.join(x[:self.top_n]) for x in ctf_labels]
            
        return ot_labels
