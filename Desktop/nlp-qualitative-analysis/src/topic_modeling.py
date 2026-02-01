"""
Topic Modeling Module for Unstructured Text Data
Supports LDA and NMF with coherence evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class TopicModeler:
    """
    Topic modeling for text data using LDA and NMF.
    
    Features:
    - Latent Dirichlet Allocation (LDA)
    - Non-negative Matrix Factorization (NMF)
    - Latent Semantic Analysis (LSA/SVD)
    - Topic coherence evaluation
    - Top terms extraction
    """
    
    def __init__(
        self,
        method: str = 'lda',
        n_topics: int = 5,
        max_features: int = 1000,
        min_df: int = 2,
        max_df: float = 0.95,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize topic modeler.
        
        Parameters:
        -----------
        method : str
            Topic modeling method: 'lda', 'nmf', or 'lsa'
        n_topics : int
            Number of topics to extract
        max_features : int
            Maximum number of features for vectorization
        min_df : int
            Minimum document frequency
        max_df : float
            Maximum document frequency (proportion)
        random_state : int
            Random seed
        **kwargs : dict
            Additional parameters for the model
        """
        self.method = method
        self.n_topics = n_topics
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.document_topic_matrix = None
        self.topic_word_matrix = None
        
    def fit(self, texts: List[str]) -> 'TopicModeler':
        """
        Fit topic model on texts.
        
        Parameters:
        -----------
        texts : list
            List of preprocessed texts
            
        Returns:
        --------
        self
        """
        # Vectorize texts
        if self.method == 'lda':
            # LDA works better with raw counts
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
        else:
            # NMF and LSA work better with TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
        
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit topic model
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=20,
                learning_method='online',
                **self.kwargs
            )
        elif self.method == 'nmf':
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=200,
                **self.kwargs
            )
        elif self.method == 'lsa':
            self.model = TruncatedSVD(
                n_components=self.n_topics,
                random_state=self.random_state,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit and transform
        self.document_topic_matrix = self.model.fit_transform(doc_term_matrix)
        self.topic_word_matrix = self.model.components_
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts to topic distributions.
        
        Parameters:
        -----------
        texts : list
            List of preprocessed texts
            
        Returns:
        --------
        np.ndarray
            Document-topic matrix
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        doc_term_matrix = self.vectorizer.transform(texts)
        return self.model.transform(doc_term_matrix)
    
    def get_top_terms(
        self,
        n_terms: int = 10,
        topic_ids: Optional[List[int]] = None
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top terms for each topic.
        
        Parameters:
        -----------
        n_terms : int
            Number of top terms to return
        topic_ids : list, optional
            Specific topic IDs to get terms for
            
        Returns:
        --------
        dict
            Dictionary mapping topic_id to list of (term, weight) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if topic_ids is None:
            topic_ids = range(self.n_topics)
        
        top_terms = {}
        
        for topic_id in topic_ids:
            # Get top term indices
            top_indices = np.argsort(self.topic_word_matrix[topic_id])[::-1][:n_terms]
            
            # Get terms and weights
            terms_weights = [
                (self.feature_names[i], self.topic_word_matrix[topic_id][i])
                for i in top_indices
            ]
            
            top_terms[topic_id] = terms_weights
        
        return top_terms
    
    def get_dominant_topic(self, doc_topic_dist: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get dominant topic for each document.
        
        Parameters:
        -----------
        doc_topic_dist : np.ndarray, optional
            Document-topic distribution matrix
            
        Returns:
        --------
        np.ndarray
            Array of dominant topic IDs
        """
        if doc_topic_dist is None:
            doc_topic_dist = self.document_topic_matrix
        
        return np.argmax(doc_topic_dist, axis=1)
    
    def get_topic_summary(self, n_terms: int = 10) -> pd.DataFrame:
        """
        Get summary of all topics.
        
        Parameters:
        -----------
        n_terms : int
            Number of top terms per topic
            
        Returns:
        --------
        pd.DataFrame
            Topic summary with top terms
        """
        top_terms = self.get_top_terms(n_terms)
        
        summaries = []
        for topic_id, terms_weights in top_terms.items():
            terms = [term for term, _ in terms_weights]
            weights = [weight for _, weight in terms_weights]
            
            summary = {
                'topic_id': topic_id,
                'top_terms': ', '.join(terms[:5]),
                'all_terms': terms,
                'weights': weights,
                'avg_weight': np.mean(weights)
            }
            
            # Add document count
            if self.document_topic_matrix is not None:
                dominant_topics = self.get_dominant_topic()
                summary['n_documents'] = (dominant_topics == topic_id).sum()
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def compute_coherence(
        self,
        texts: List[str],
        method: str = 'c_v',
        n_terms: int = 10
    ) -> float:
        """
        Compute topic coherence score.
        
        Parameters:
        -----------
        texts : list
            Original texts
        method : str
            Coherence method: 'c_v', 'u_mass', or 'npmi'
        n_terms : int
            Number of top terms to use
            
        Returns:
        --------
        float
            Coherence score
        """
        top_terms = self.get_top_terms(n_terms)
        
        # Simple coherence based on term co-occurrence
        # For production, consider using gensim's CoherenceModel
        coherence_scores = []
        
        for topic_id, terms_weights in top_terms.items():
            terms = [term for term, _ in terms_weights]
            
            # Count co-occurrences
            co_occur = 0
            total_pairs = 0
            
            for i, term1 in enumerate(terms):
                for term2 in terms[i+1:]:
                    # Check if both terms appear in same documents
                    term1_docs = sum(1 for text in texts if term1 in text.lower())
                    term2_docs = sum(1 for text in texts if term2 in text.lower())
                    both_docs = sum(
                        1 for text in texts
                        if term1 in text.lower() and term2 in text.lower()
                    )
                    
                    if term1_docs > 0 and term2_docs > 0:
                        # NPMI-like score
                        if both_docs > 0:
                            pmi = np.log((both_docs * len(texts)) / (term1_docs * term2_docs))
                            npmi = pmi / -np.log(both_docs / len(texts))
                            co_occur += npmi
                        total_pairs += 1
            
            if total_pairs > 0:
                coherence_scores.append(co_occur / total_pairs)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def find_optimal_topics(
        self,
        texts: List[str],
        topic_range: range = range(2, 11),
        metric: str = 'coherence'
    ) -> Tuple[int, Dict]:
        """
        Find optimal number of topics.
        
        Parameters:
        -----------
        texts : list
            List of texts
        topic_range : range
            Range of topic numbers to test
        metric : str
            Metric to optimize: 'coherence' or 'perplexity'
            
        Returns:
        --------
        tuple
            (optimal_n_topics, scores_dict)
        """
        scores = {'n_topics': [], 'score': []}
        
        for n in topic_range:
            # Fit model
            modeler = TopicModeler(
                method=self.method,
                n_topics=n,
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                random_state=self.random_state
            )
            modeler.fit(texts)
            
            scores['n_topics'].append(n)
            
            if metric == 'coherence':
                score = modeler.compute_coherence(texts)
                scores['score'].append(score)
            elif metric == 'perplexity' and self.method == 'lda':
                # LDA has perplexity
                doc_term_matrix = modeler.vectorizer.transform(texts)
                score = modeler.model.perplexity(doc_term_matrix)
                scores['score'].append(score)
        
        # Find optimal
        if metric == 'coherence':
            optimal_idx = np.argmax(scores['score'])
        else:
            optimal_idx = np.argmin(scores['score'])
        
        optimal_n_topics = scores['n_topics'][optimal_idx]
        
        return optimal_n_topics, scores

# Made with Bob
