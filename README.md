# Exhaustive HDBSCAN

HDBSCAN is a density based clustering technique that results in stable clusters. It is a modification of the basic DBSCAN algorithm and the hierarchical element comes from the splitting technique that proceeds in order of the density metric, i.e. the splits are performed as the density thresholds are reached sequentially. Consequently points within a threshold that do not qualify the cluster size criteria, are qualified as noise.

Since outputs of HDBSCAN clustering leaves out some points as noise, this can lead to loss of information. Exhaustive HDBSCAN addresses this shortfall by iteratively clustering the left over noise points at each level of iteration. The points left out as noise at each iteration are combined into their own density structure and clustered through their unique thresholding. 

Typically, HDBSCAN is combined with a dimension reduction technique. Universal Manifold Approximation and Projection (UMAP) is more suited to the HDBSCAN as it preserves local neighborhood topology. This preserves the density structure while solidifying the separation boundaries. 

With Exhaustive HDBSCAN, a new reduction is performed at each iteration. This leads to noise points forming their own unique density structure within which thresholding can separate clusters. Cluster size and minimum samples are adjusted iteratively to account for the reduced dataset size as iterations proceed.

Further when noise points from each iteration are reduced to form their unique density structure, these clusters can now be linked back to the clusters they originally fell out of. Exhaustive HDBSCAN caters for this by using a HDBSCAN-like density-based metric to assign parent-child links between clusters from previous iteration and current iteration. The resultant output is a tree of clusters.

Tests have found that as we move down this tree of clusters, the clusters become more sparse, but the meaning within them soldifies and becomes more precise.

Exhaustive HDBSCAN becomes an effective tool to extract structured meaning from large datasets in an unsupervised manner.

[Full Docs](https://smudgeapp.github.io/exhaustive_hdbscan/)

## Installation

	pip install exhaustive_hdbscan
	
## Basic Usage

Here *input_data* is raw feature set.

	from exhaustive_hdbscan import EHDBSCAN
	
	ehdb = EHDBSCAN()
	ehdb = ehdb.fit(input_data)
	
	print(ehdb.cluster_feats.parent_names)
	
## Minimal Reducer and Encoder Implementation

	from sentence_transformers import SentenceTransformer
	import umap
	from exhaustive_hdbscan import Reducer, Encoder

	class mEncoder(Encoder):
		def __init__(self):
			self.encoder = SentenceTransformer('all-mpnet-base-v2')

		def encode(self, X):
			embed = self.encoder.encode(X)
			embed = normalize(embed)
			return embed

	class mReducer(Reducer):
		def __init__(self):
			super().__init__()
			self.ump = umap.UMAP(metric='cosine', random_state=200585)

		def fit(self, X):
			super().__init__()
			self.ump.fit(X)
			return self.ump.embedding_

		def transform(self, X):
			super().__init__()
			return self.ump.transform(X)