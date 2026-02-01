"""
Theme Analysis Module for UX Research
Advanced theme stability, robustness, and overlap analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from scipy.stats import entropy
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')


class ThemeAnalyzer:
    """
    Advanced theme analysis for qualitative research.
    
    Features:
    - Theme stability across resamples
    - Theme coherence and strength indicators
    - Theme overlap and boundary detection
    - Emergent vs dominant theme identification
    - Sensitivity analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize theme analyzer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.theme_stability_results = {}
        self.theme_overlaps = {}
        
    def compute_theme_stability(
        self,
        texts: List[str],
        vectorizer,
        model,
        n_resamples: int = 10,
        sample_fraction: float = 0.8
    ) -> Dict:
        """
        Compute theme stability across bootstrap resamples.
        
        Parameters:
        -----------
        texts : List[str]
            Input texts
        vectorizer : fitted vectorizer
            TF-IDF or Count vectorizer
        model : fitted topic model
            LDA, NMF, or LSA model
        n_resamples : int
            Number of bootstrap resamples
        sample_fraction : float
            Fraction of data to sample each iteration
            
        Returns:
        --------
        Dict with stability metrics per theme
        """
        np.random.seed(self.random_state)
        n_samples = len(texts)
        n_topics = model.n_components if hasattr(model, 'n_components') else model.n_topics
        
        # Store topic-term distributions for each resample
        topic_distributions = []
        
        for i in range(n_resamples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=int(n_samples * sample_fraction), replace=True)
            sample_texts = [texts[idx] for idx in indices]
            
            # Refit model
            X_sample = vectorizer.transform(sample_texts)
            
            # Clone and fit model
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_sample)
            
            # Get topic-term distribution
            if hasattr(model_clone, 'components_'):
                topic_dist = model_clone.components_
            else:
                topic_dist = model_clone.topic_word_
                
            topic_distributions.append(topic_dist)
        
        # Compute stability metrics
        stability_scores = {}
        
        for topic_id in range(n_topics):
            # Get this topic's distribution across resamples
            topic_vectors = [dist[topic_id] for dist in topic_distributions]
            
            # Compute pairwise similarities
            similarities = []
            for i in range(len(topic_vectors)):
                for j in range(i + 1, len(topic_vectors)):
                    sim = cosine_similarity(
                        topic_vectors[i].reshape(1, -1),
                        topic_vectors[j].reshape(1, -1)
                    )[0, 0]
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            std_similarity = np.std(similarities) if similarities else 0.0
            
            # Stability score (0-1, higher is more stable)
            stability = avg_similarity
            
            # Robustness indicator
            if stability >= 0.85:
                robustness = "Very Robust"
            elif stability >= 0.70:
                robustness = "Robust"
            elif stability >= 0.50:
                robustness = "Moderately Stable"
            else:
                robustness = "Unstable"
            
            stability_scores[topic_id] = {
                'stability_score': stability,
                'stability_std': std_similarity,
                'robustness_label': robustness,
                'persistence_rate': stability * 100,  # As percentage
                'confidence': 'High' if stability >= 0.70 else 'Medium' if stability >= 0.50 else 'Low'
            }
        
        self.theme_stability_results = stability_scores
        return stability_scores
    
    def compute_theme_coherence(
        self,
        topic_model,
        vectorizer,
        texts: List[str],
        n_top_words: int = 10
    ) -> Dict:
        """
        Compute theme coherence scores (internal consistency).
        
        Parameters:
        -----------
        topic_model : fitted topic model
            LDA, NMF, or LSA model
        vectorizer : fitted vectorizer
            TF-IDF or Count vectorizer
        texts : List[str]
            Input texts
        n_top_words : int
            Number of top words to consider
            
        Returns:
        --------
        Dict with coherence scores per theme
        """
        feature_names = vectorizer.get_feature_names_out()
        n_topics = topic_model.n_components if hasattr(topic_model, 'n_components') else topic_model.n_topics
        
        coherence_scores = {}
        
        for topic_id in range(n_topics):
            # Get top words for this topic
            if hasattr(topic_model, 'components_'):
                topic_dist = topic_model.components_[topic_id]
            else:
                topic_dist = topic_model.topic_word_[topic_id]
            
            top_indices = topic_dist.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            
            # Compute coherence using PMI-like measure
            # Simplified coherence: average pairwise word co-occurrence
            coherence = self._compute_word_coherence(top_words, texts)
            
            # Theme strength label
            if coherence >= 0.7:
                strength = "Strong"
            elif coherence >= 0.5:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            coherence_scores[topic_id] = {
                'coherence_score': coherence,
                'theme_strength': strength,
                'top_words': top_words,
                'interpretability': 'High' if coherence >= 0.6 else 'Medium' if coherence >= 0.4 else 'Low'
            }
        
        return coherence_scores
    
    def _compute_word_coherence(self, words: List[str], texts: List[str]) -> float:
        """
        Compute coherence score for a set of words.
        Simplified PMI-based coherence.
        """
        # Count word occurrences and co-occurrences
        word_counts = Counter()
        cooccurrence_counts = defaultdict(int)
        
        for text in texts:
            text_lower = text.lower()
            words_in_text = [w for w in words if w in text_lower]
            
            for word in words_in_text:
                word_counts[word] += 1
            
            # Count co-occurrences
            for i, w1 in enumerate(words_in_text):
                for w2 in words_in_text[i+1:]:
                    pair = tuple(sorted([w1, w2]))
                    cooccurrence_counts[pair] += 1
        
        # Compute coherence
        if len(word_counts) < 2:
            return 0.0
        
        coherence_scores = []
        n_docs = len(texts)
        
        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                pair = tuple(sorted([w1, w2]))
                
                p_w1 = word_counts.get(w1, 0) / n_docs
                p_w2 = word_counts.get(w2, 0) / n_docs
                p_w1_w2 = cooccurrence_counts.get(pair, 0) / n_docs
                
                if p_w1 > 0 and p_w2 > 0 and p_w1_w2 > 0:
                    # PMI-like score
                    score = p_w1_w2 / (p_w1 * p_w2)
                    coherence_scores.append(min(score, 1.0))
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def compute_theme_overlap(
        self,
        document_topic_matrix: np.ndarray,
        threshold: float = 0.1
    ) -> Dict:
        """
        Compute theme overlap and boundary detection.
        
        Parameters:
        -----------
        document_topic_matrix : np.ndarray
            Document-topic distribution matrix (n_docs x n_topics)
        threshold : float
            Minimum probability to consider a document belongs to a theme
            
        Returns:
        --------
        Dict with overlap metrics and boundary documents
        """
        n_docs, n_topics = document_topic_matrix.shape
        
        # Normalize to probabilities
        doc_topic_probs = document_topic_matrix / document_topic_matrix.sum(axis=1, keepdims=True)
        
        # Compute overlap matrix (theme-theme similarity)
        overlap_matrix = np.zeros((n_topics, n_topics))
        
        for i in range(n_topics):
            for j in range(i, n_topics):
                # Count documents that belong to both themes
                docs_in_i = doc_topic_probs[:, i] >= threshold
                docs_in_j = doc_topic_probs[:, j] >= threshold
                overlap = np.sum(docs_in_i & docs_in_j)
                
                # Jaccard similarity
                union = np.sum(docs_in_i | docs_in_j)
                similarity = overlap / union if union > 0 else 0.0
                
                overlap_matrix[i, j] = similarity
                overlap_matrix[j, i] = similarity
        
        # Find boundary documents (high entropy across themes)
        doc_entropies = entropy(doc_topic_probs.T)
        
        # Identify fuzzy assignments
        max_probs = doc_topic_probs.max(axis=1)
        fuzzy_docs = np.where(max_probs < 0.5)[0]
        
        # Fuzziness indicator per theme
        theme_fuzziness = {}
        for topic_id in range(n_topics):
            docs_in_theme = np.where(doc_topic_probs[:, topic_id] >= threshold)[0]
            if len(docs_in_theme) > 0:
                avg_max_prob = max_probs[docs_in_theme].mean()
                fuzziness = 1 - avg_max_prob
                
                theme_fuzziness[topic_id] = {
                    'fuzziness_score': fuzziness,
                    'clarity': 'Clear' if fuzziness < 0.3 else 'Moderate' if fuzziness < 0.5 else 'Fuzzy',
                    'n_boundary_docs': len(set(docs_in_theme) & set(fuzzy_docs)),
                    'n_total_docs': len(docs_in_theme)
                }
        
        self.theme_overlaps = {
            'overlap_matrix': overlap_matrix,
            'boundary_documents': fuzzy_docs.tolist(),
            'document_entropies': doc_entropies,
            'theme_fuzziness': theme_fuzziness
        }
        
        return self.theme_overlaps
    
    def identify_emergent_themes(
        self,
        document_topic_matrix: np.ndarray,
        sentiment_scores: Optional[np.ndarray] = None,
        frequency_threshold: float = 0.1,
        intensity_threshold: float = 0.7
    ) -> Dict:
        """
        Identify emergent vs dominant themes.
        
        Parameters:
        -----------
        document_topic_matrix : np.ndarray
            Document-topic distribution
        sentiment_scores : np.ndarray, optional
            Sentiment intensity scores per document
        frequency_threshold : float
            Threshold for low-frequency themes
        intensity_threshold : float
            Threshold for high-intensity themes
            
        Returns:
        --------
        Dict categorizing themes as dominant, emergent, or weak signals
        """
        n_docs, n_topics = document_topic_matrix.shape
        
        # Compute frequency (prevalence)
        topic_frequencies = document_topic_matrix.sum(axis=0) / n_docs
        
        # Compute intensity (average strength when present)
        topic_intensities = []
        for topic_id in range(n_topics):
            topic_probs = document_topic_matrix[:, topic_id]
            # Only consider documents where this topic is present
            present_docs = topic_probs > 0.01
            if present_docs.sum() > 0:
                avg_intensity = topic_probs[present_docs].mean()
            else:
                avg_intensity = 0.0
            topic_intensities.append(avg_intensity)
        
        topic_intensities = np.array(topic_intensities)
        
        # Compute emotional weight if sentiment provided
        emotional_weights = None
        if sentiment_scores is not None:
            emotional_weights = []
            for topic_id in range(n_topics):
                topic_probs = document_topic_matrix[:, topic_id]
                # Weight sentiment by topic probability
                weighted_sentiment = np.abs(sentiment_scores) * topic_probs
                avg_emotional_weight = weighted_sentiment.sum() / (topic_probs.sum() + 1e-10)
                emotional_weights.append(avg_emotional_weight)
            emotional_weights = np.array(emotional_weights)
        
        # Categorize themes
        theme_categories = {}
        
        for topic_id in range(n_topics):
            freq = topic_frequencies[topic_id]
            intensity = topic_intensities[topic_id]
            
            # Categorization logic
            if freq >= frequency_threshold and intensity >= intensity_threshold:
                category = "Dominant Theme"
                priority = "High"
            elif freq < frequency_threshold and intensity >= intensity_threshold:
                category = "Emergent Theme (Weak Signal)"
                priority = "High"
            elif freq >= frequency_threshold and intensity < intensity_threshold:
                category = "Background Theme"
                priority = "Medium"
            else:
                category = "Noise"
                priority = "Low"
            
            theme_categories[topic_id] = {
                'category': category,
                'priority': priority,
                'frequency': freq,
                'intensity': intensity,
                'prevalence_pct': freq * 100,
                'emotional_weight': emotional_weights[topic_id] if emotional_weights is not None else None
            }
        
        return theme_categories
    
    def get_theme_summary(self) -> pd.DataFrame:
        """
        Get comprehensive theme summary combining all metrics.
        
        Returns:
        --------
        DataFrame with all theme metrics
        """
        if not self.theme_stability_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for topic_id, metrics in self.theme_stability_results.items():
            row = {
                'theme_id': topic_id,
                'stability_score': metrics['stability_score'],
                'robustness': metrics['robustness_label'],
                'confidence': metrics['confidence']
            }
            
            # Add overlap metrics if available
            if self.theme_overlaps and 'theme_fuzziness' in self.theme_overlaps:
                if topic_id in self.theme_overlaps['theme_fuzziness']:
                    fuzz = self.theme_overlaps['theme_fuzziness'][topic_id]
                    row['clarity'] = fuzz['clarity']
                    row['fuzziness'] = fuzz['fuzziness_score']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

# Made with Bob
