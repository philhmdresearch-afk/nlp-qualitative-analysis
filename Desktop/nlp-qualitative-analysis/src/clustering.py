"""
Clustering Module for Both Structured and Unstructured Data
Supports multiple algorithms with energy-efficient implementations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering,
    DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings('ignore')

# Lazy imports for optional dependencies
_hdbscan_available = None
_kmodes_available = None


def _check_hdbscan():
    """Check if HDBSCAN is available."""
    global _hdbscan_available
    if _hdbscan_available is None:
        try:
            import hdbscan
            _hdbscan_available = True
        except ImportError:
            _hdbscan_available = False
    return _hdbscan_available


def _check_kmodes():
    """Check if kmodes is available."""
    global _kmodes_available
    if _kmodes_available is None:
        try:
            from kmodes.kmodes import KModes
            from kmodes.kprototypes import KPrototypes
            _kmodes_available = True
        except ImportError:
            _kmodes_available = False
    return _kmodes_available


class ClusterAnalyzer:
    """
    Unified clustering analysis for structured and unstructured data.
    
    Supported algorithms:
    - K-means (standard and mini-batch)
    - Gaussian Mixture Models
    - Hierarchical Clustering (Ward, complete, average)
    - HDBSCAN (density-based)
    - K-modes/K-prototypes (categorical data)
    - Spherical K-means (text/TF-IDF data)
    """
    
    def __init__(
        self,
        algorithm: str = 'kmeans',
        n_clusters: int = 3,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize cluster analyzer.
        
        Parameters:
        -----------
        algorithm : str
            Clustering algorithm: 'kmeans', 'minibatch_kmeans', 'gmm',
            'hierarchical', 'hdbscan', 'kmodes', 'kprototypes', 'spherical_kmeans'
        n_clusters : int
            Number of clusters (not used for HDBSCAN)
        random_state : int
            Random seed for reproducibility
        **kwargs : dict
            Additional algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.metrics_ = {}
        
    def fit(self, X: np.ndarray, **fit_kwargs) -> 'ClusterAnalyzer':
        """
        Fit clustering model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        **fit_kwargs : dict
            Additional fit parameters
            
        Returns:
        --------
        self
        """
        if self.algorithm == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                **self.kwargs
            )
            self.model.fit(X)
            self.labels_ = self.model.labels_
            self.cluster_centers_ = self.model.cluster_centers_
            
        elif self.algorithm == 'minibatch_kmeans':
            self.model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=1000,
                **self.kwargs
            )
            self.model.fit(X)
            self.labels_ = self.model.labels_
            self.cluster_centers_ = self.model.cluster_centers_
            
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs
            )
            self.model.fit(X)
            self.labels_ = self.model.predict(X)
            self.cluster_centers_ = self.model.means_
            
        elif self.algorithm == 'hierarchical':
            linkage_method = self.kwargs.get('linkage', 'ward')
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=linkage_method,
                **{k: v for k, v in self.kwargs.items() if k != 'linkage'}
            )
            self.labels_ = self.model.fit_predict(X)
            # Compute cluster centers manually
            self.cluster_centers_ = self._compute_centers(X, self.labels_)
            
        elif self.algorithm == 'hdbscan':
            if not _check_hdbscan():
                raise ImportError("HDBSCAN not installed. Install with: pip install hdbscan")
            import hdbscan
            
            min_cluster_size = self.kwargs.get('min_cluster_size', 5)
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                **{k: v for k, v in self.kwargs.items() if k != 'min_cluster_size'}
            )
            self.labels_ = self.model.fit_predict(X)
            # Filter out noise points (-1 label)
            valid_labels = self.labels_[self.labels_ >= 0]
            if len(valid_labels) > 0:
                self.n_clusters = len(np.unique(valid_labels))
                self.cluster_centers_ = self._compute_centers(X, self.labels_)
            
        elif self.algorithm == 'kmodes':
            if not _check_kmodes():
                raise ImportError("kmodes not installed. Install with: pip install kmodes")
            from kmodes.kmodes import KModes
            
            self.model = KModes(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs
            )
            self.labels_ = self.model.fit_predict(X)
            self.cluster_centers_ = self.model.cluster_centroids_
            
        elif self.algorithm == 'kprototypes':
            if not _check_kmodes():
                raise ImportError("kmodes not installed. Install with: pip install kmodes")
            from kmodes.kprototypes import KPrototypes
            
            categorical_indices = fit_kwargs.get('categorical', [])
            self.model = KPrototypes(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs
            )
            self.labels_ = self.model.fit_predict(X, categorical=categorical_indices)
            self.cluster_centers_ = self.model.cluster_centroids_
            
        elif self.algorithm == 'spherical_kmeans':
            # Spherical K-means for text data (cosine similarity)
            self._fit_spherical_kmeans(X)
            
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Compute evaluation metrics
        self._compute_metrics(X)
        
        return self
    
    def _fit_spherical_kmeans(self, X: np.ndarray):
        """Fit spherical K-means (for text/TF-IDF data)."""
        from sklearn.preprocessing import normalize
        
        # Normalize vectors to unit length
        X_normalized = normalize(X, norm='l2')
        
        # Use standard K-means on normalized data
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            **self.kwargs
        )
        self.model.fit(X_normalized)
        self.labels_ = self.model.labels_
        
        # Normalize cluster centers
        self.cluster_centers_ = normalize(self.model.cluster_centers_, norm='l2')
    
    def _compute_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centers manually."""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            if label >= 0:  # Skip noise points
                mask = labels == label
                center = X[mask].mean(axis=0)
                centers.append(center)
        
        return np.array(centers) if centers else np.array([])
    
    def _compute_metrics(self, X: np.ndarray):
        """Compute clustering evaluation metrics."""
        if self.labels_ is None or len(np.unique(self.labels_)) < 2:
            return
        
        # Filter out noise points for HDBSCAN
        valid_mask = self.labels_ >= 0
        X_valid = X[valid_mask]
        labels_valid = self.labels_[valid_mask]
        
        if len(np.unique(labels_valid)) < 2:
            return
        
        try:
            self.metrics_['silhouette'] = silhouette_score(X_valid, labels_valid)
        except:
            self.metrics_['silhouette'] = None
        
        try:
            self.metrics_['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
        except:
            self.metrics_['davies_bouldin'] = None
        
        try:
            self.metrics_['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
        except:
            self.metrics_['calinski_harabasz'] = None
        
        # Cluster sizes
        unique, counts = np.unique(labels_valid, return_counts=True)
        self.metrics_['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            # For algorithms without predict (e.g., hierarchical)
            # Use nearest cluster center
            from scipy.spatial.distance import cdist
            distances = cdist(X, self.cluster_centers_)
            return np.argmin(distances, axis=1)
    
    def get_cluster_summary(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        pd.DataFrame
            Cluster summary statistics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summaries = []
        
        for cluster_id in np.unique(self.labels_):
            if cluster_id < 0:  # Skip noise points
                continue
            
            mask = self.labels_ == cluster_id
            cluster_data = X[mask]
            
            summary = {
                'cluster': cluster_id,
                'size': mask.sum(),
                'percentage': mask.sum() / len(X) * 100
            }
            
            # Feature statistics
            if feature_names:
                for i, name in enumerate(feature_names):
                    summary[f'{name}_mean'] = cluster_data[:, i].mean()
                    summary[f'{name}_std'] = cluster_data[:, i].std()
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11),
        method: str = 'elbow'
    ) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        k_range : range
            Range of k values to test
        method : str
            Method: 'elbow' (inertia), 'silhouette', or 'both'
            
        Returns:
        --------
        tuple
            (optimal_k, scores_dict)
        """
        scores = {'k': [], 'inertia': [], 'silhouette': []}
        
        for k in k_range:
            # Fit model
            if self.algorithm in ['kmeans', 'minibatch_kmeans', 'spherical_kmeans']:
                model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                model.fit(X)
                labels = model.labels_
                
                scores['k'].append(k)
                scores['inertia'].append(model.inertia_)
                
                if len(np.unique(labels)) > 1:
                    scores['silhouette'].append(silhouette_score(X, labels))
                else:
                    scores['silhouette'].append(0)
        
        # Determine optimal k
        if method == 'elbow':
            # Use elbow method (find point of maximum curvature)
            inertias = np.array(scores['inertia'])
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            optimal_k = scores['k'][np.argmin(diffs2) + 1]
        elif method == 'silhouette':
            # Use maximum silhouette score
            optimal_k = scores['k'][np.argmax(scores['silhouette'])]
        else:
            # Use both methods and take average
            elbow_k = scores['k'][np.argmin(np.diff(np.diff(scores['inertia']))) + 1]
            silhouette_k = scores['k'][np.argmax(scores['silhouette'])]
            optimal_k = int((elbow_k + silhouette_k) / 2)
        
        return optimal_k, scores


def compute_hierarchical_linkage(
    X: np.ndarray,
    method: str = 'ward',
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute linkage matrix for hierarchical clustering.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    method : str
        Linkage method: 'ward', 'complete', 'average', 'single'
    metric : str
        Distance metric
        
    Returns:
    --------
    np.ndarray
        Linkage matrix for dendrogram
    """
    if method == 'ward':
        # Ward requires euclidean distance
        return linkage(X, method='ward')
    else:
        # Compute distance matrix first
        distances = pdist(X, metric=metric)
        return linkage(distances, method=method)


def create_dendrogram_data(
    linkage_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    truncate_mode: Optional[str] = None,
    p: int = 30
) -> Dict:
    """
    Create dendrogram data structure.
    
    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    labels : list, optional
        Sample labels
    truncate_mode : str, optional
        Truncation mode: 'lastp', 'level', None
    p : int
        Truncation parameter
        
    Returns:
    --------
    dict
        Dendrogram data
    """
    dend = dendrogram(
        linkage_matrix,
        labels=labels,
        truncate_mode=truncate_mode,
        p=p,
        no_plot=True
    )
    
    return dend

# Made with Bob
