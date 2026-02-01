"""
Contrastive Analysis Module for UX Research
Compare themes, sentiment, and patterns across groups
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from scipy.stats import chi2_contingency, mannwhitneyu
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')


class ContrastiveAnalyzer:
    """
    Contrastive analysis for comparing groups in qualitative research.
    
    Features:
    - Contrastive theme summaries
    - Theme × Sentiment interactions
    - Group-specific theme patterns
    - Natural language contrast statements
    """
    
    def __init__(self):
        """Initialize contrastive analyzer."""
        self.contrast_results = {}
        
    def generate_contrastive_summary(
        self,
        texts: List[str],
        groups: List[str],
        document_topic_matrix: np.ndarray,
        topic_terms: Dict[int, List[Tuple[str, float]]],
        group_a: str,
        group_b: str
    ) -> Dict:
        """
        Generate contrastive theme summaries between two groups.
        
        Parameters:
        -----------
        texts : List[str]
            Original texts
        groups : List[str]
            Group labels for each text
        document_topic_matrix : np.ndarray
            Document-topic distribution
        topic_terms : Dict
            Top terms for each topic
        group_a : str
            First group identifier
        group_b : str
            Second group identifier
            
        Returns:
        --------
        Dict with contrastive summaries
        """
        # Split data by groups
        group_a_mask = np.array([g == group_a for g in groups])
        group_b_mask = np.array([g == group_b for g in groups])
        
        group_a_topics = document_topic_matrix[group_a_mask]
        group_b_topics = document_topic_matrix[group_b_mask]
        
        n_topics = document_topic_matrix.shape[1]
        
        # Compute topic prevalence in each group
        group_a_prevalence = group_a_topics.mean(axis=0)
        group_b_prevalence = group_b_topics.mean(axis=0)
        
        # Compute differences
        prevalence_diff = group_a_prevalence - group_b_prevalence
        
        # Identify distinctive themes
        distinctive_a = []
        distinctive_b = []
        shared_themes = []
        
        for topic_id in range(n_topics):
            diff = prevalence_diff[topic_id]
            
            # Get top terms for this topic
            terms = [term for term, _ in topic_terms.get(topic_id, [])][:5]
            terms_str = ", ".join(terms)
            
            if abs(diff) < 0.05:  # Shared theme
                shared_themes.append({
                    'topic_id': topic_id,
                    'terms': terms_str,
                    'group_a_prevalence': group_a_prevalence[topic_id],
                    'group_b_prevalence': group_b_prevalence[topic_id],
                    'difference': diff
                })
            elif diff > 0.05:  # More prevalent in group A
                distinctive_a.append({
                    'topic_id': topic_id,
                    'terms': terms_str,
                    'prevalence': group_a_prevalence[topic_id],
                    'difference': diff,
                    'relative_strength': diff / (group_b_prevalence[topic_id] + 0.01)
                })
            else:  # More prevalent in group B
                distinctive_b.append({
                    'topic_id': topic_id,
                    'terms': terms_str,
                    'prevalence': group_b_prevalence[topic_id],
                    'difference': abs(diff),
                    'relative_strength': abs(diff) / (group_a_prevalence[topic_id] + 0.01)
                })
        
        # Sort by difference magnitude
        distinctive_a.sort(key=lambda x: x['difference'], reverse=True)
        distinctive_b.sort(key=lambda x: x['difference'], reverse=True)
        
        # Generate natural language summaries
        summaries = self._generate_contrast_statements(
            group_a, group_b,
            distinctive_a, distinctive_b, shared_themes
        )
        
        return {
            'group_a': group_a,
            'group_b': group_b,
            'distinctive_to_a': distinctive_a,
            'distinctive_to_b': distinctive_b,
            'shared_themes': shared_themes,
            'natural_language_summaries': summaries
        }
    
    def _generate_contrast_statements(
        self,
        group_a: str,
        group_b: str,
        distinctive_a: List[Dict],
        distinctive_b: List[Dict],
        shared_themes: List[Dict]
    ) -> List[str]:
        """Generate natural language contrast statements."""
        statements = []
        
        # Distinctive to group A
        if distinctive_a:
            top_theme = distinctive_a[0]
            statements.append(
                f"{group_a} discusses {top_theme['terms']} significantly more than {group_b} "
                f"({top_theme['prevalence']:.1%} vs {top_theme['prevalence'] - top_theme['difference']:.1%})"
            )
        
        # Distinctive to group B
        if distinctive_b:
            top_theme = distinctive_b[0]
            statements.append(
                f"{group_b} focuses more on {top_theme['terms']} compared to {group_a} "
                f"({top_theme['prevalence']:.1%} vs {top_theme['prevalence'] - top_theme['difference']:.1%})"
            )
        
        # Shared themes with different emphasis
        if shared_themes:
            for theme in shared_themes[:2]:  # Top 2 shared themes
                if abs(theme['difference']) > 0.02:
                    if theme['difference'] > 0:
                        statements.append(
                            f"Both groups discuss {theme['terms']}, but {group_a} emphasizes it slightly more"
                        )
                    else:
                        statements.append(
                            f"Both groups discuss {theme['terms']}, but {group_b} emphasizes it slightly more"
                        )
        
        # Summary statement
        if distinctive_a and distinctive_b:
            statements.append(
                f"Key difference: {group_a} prioritizes {distinctive_a[0]['terms']}, "
                f"while {group_b} prioritizes {distinctive_b[0]['terms']}"
            )
        
        return statements
    
    def analyze_theme_sentiment_interaction(
        self,
        document_topic_matrix: np.ndarray,
        sentiment_scores: np.ndarray,
        groups: List[str],
        topic_terms: Dict[int, List[Tuple[str, float]]]
    ) -> Dict:
        """
        Analyze Theme × Sentiment interactions across groups.
        
        Parameters:
        -----------
        document_topic_matrix : np.ndarray
            Document-topic distribution
        sentiment_scores : np.ndarray
            Sentiment compound scores per document
        groups : List[str]
            Group labels
        topic_terms : Dict
            Top terms for each topic
            
        Returns:
        --------
        Dict with theme-sentiment interactions
        """
        n_topics = document_topic_matrix.shape[1]
        unique_groups = sorted(set(groups))
        
        interactions = {}
        
        for topic_id in range(n_topics):
            topic_probs = document_topic_matrix[:, topic_id]
            
            # Get documents where this topic is prominent (>0.1)
            prominent_docs = topic_probs > 0.1
            
            if prominent_docs.sum() < 5:  # Skip if too few documents
                continue
            
            # Compute sentiment for this topic across groups
            group_sentiments = {}
            
            for group in unique_groups:
                group_mask = np.array([g == group for g in groups])
                topic_group_mask = prominent_docs & group_mask
                
                if topic_group_mask.sum() > 0:
                    # Weight sentiment by topic probability
                    weighted_sentiment = (
                        sentiment_scores[topic_group_mask] * 
                        topic_probs[topic_group_mask]
                    ).sum() / topic_probs[topic_group_mask].sum()
                    
                    group_sentiments[group] = {
                        'avg_sentiment': weighted_sentiment,
                        'n_docs': topic_group_mask.sum(),
                        'sentiment_label': self._sentiment_label(weighted_sentiment)
                    }
            
            # Get top terms
            terms = [term for term, _ in topic_terms.get(topic_id, [])][:5]
            
            interactions[topic_id] = {
                'topic_terms': ", ".join(terms),
                'group_sentiments': group_sentiments,
                'sentiment_variance': np.var([s['avg_sentiment'] for s in group_sentiments.values()]),
                'polarity_shift': self._detect_polarity_shift(group_sentiments)
            }
        
        return interactions
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def _detect_polarity_shift(self, group_sentiments: Dict) -> Optional[str]:
        """Detect if there's a polarity shift across groups."""
        sentiments = [s['avg_sentiment'] for s in group_sentiments.values()]
        
        if len(sentiments) < 2:
            return None
        
        # Check if sentiments have opposite signs
        has_positive = any(s > 0.05 for s in sentiments)
        has_negative = any(s < -0.05 for s in sentiments)
        
        if has_positive and has_negative:
            return "Polarity Shift Detected"
        
        return None
    
    def create_salience_emotional_plot_data(
        self,
        document_topic_matrix: np.ndarray,
        sentiment_scores: np.ndarray,
        topic_terms: Dict[int, List[Tuple[str, float]]]
    ) -> pd.DataFrame:
        """
        Create data for salience vs emotional weight scatterplot.
        
        Parameters:
        -----------
        document_topic_matrix : np.ndarray
            Document-topic distribution
        sentiment_scores : np.ndarray
            Sentiment scores
        topic_terms : Dict
            Top terms for each topic
            
        Returns:
        --------
        DataFrame with salience and emotional weight per theme
        """
        n_topics = document_topic_matrix.shape[1]
        
        plot_data = []
        
        for topic_id in range(n_topics):
            topic_probs = document_topic_matrix[:, topic_id]
            
            # Salience (prevalence)
            salience = topic_probs.mean()
            
            # Emotional weight (sentiment intensity weighted by topic probability)
            emotional_weight = np.abs(sentiment_scores * topic_probs).sum() / (topic_probs.sum() + 1e-10)
            
            # Sentiment variance (emotional volatility)
            docs_with_topic = topic_probs > 0.05
            if docs_with_topic.sum() > 1:
                sentiment_variance = np.var(sentiment_scores[docs_with_topic])
            else:
                sentiment_variance = 0.0
            
            # Get top terms
            terms = [term for term, _ in topic_terms.get(topic_id, [])][:3]
            label = ", ".join(terms)
            
            # Categorize into quadrants
            if salience >= 0.15 and emotional_weight >= 0.3:
                quadrant = "Loud & Emotional"
                priority = "Critical"
            elif salience < 0.15 and emotional_weight >= 0.3:
                quadrant = "Quiet but Painful"
                priority = "High"
            elif salience >= 0.15 and emotional_weight < 0.3:
                quadrant = "Core Strengths"
                priority = "Medium"
            else:
                quadrant = "Background Noise"
                priority = "Low"
            
            plot_data.append({
                'topic_id': topic_id,
                'label': label,
                'salience': salience,
                'emotional_weight': emotional_weight,
                'sentiment_variance': sentiment_variance,
                'quadrant': quadrant,
                'priority': priority
            })
        
        return pd.DataFrame(plot_data)
    
    def compare_theme_framings(
        self,
        texts: List[str],
        groups: List[str],
        document_topic_matrix: np.ndarray,
        topic_id: int,
        n_examples: int = 3
    ) -> Dict:
        """
        Compare how different groups frame the same theme.
        
        Parameters:
        -----------
        texts : List[str]
            Original texts
        groups : List[str]
            Group labels
        document_topic_matrix : np.ndarray
            Document-topic distribution
        topic_id : int
            Topic to analyze
        n_examples : int
            Number of example quotes per group
            
        Returns:
        --------
        Dict with framing comparison
        """
        topic_probs = document_topic_matrix[:, topic_id]
        unique_groups = sorted(set(groups))
        
        framings = {}
        
        for group in unique_groups:
            group_mask = np.array([g == group for g in groups])
            group_topic_probs = topic_probs[group_mask]
            group_texts = [texts[i] for i, m in enumerate(group_mask) if m]
            
            # Get top documents for this topic in this group
            top_indices = group_topic_probs.argsort()[-n_examples:][::-1]
            
            examples = []
            for idx in top_indices:
                if idx < len(group_texts):
                    examples.append({
                        'text': group_texts[idx][:200] + "..." if len(group_texts[idx]) > 200 else group_texts[idx],
                        'topic_strength': float(group_topic_probs[idx])
                    })
            
            framings[group] = {
                'examples': examples,
                'avg_topic_strength': float(group_topic_probs.mean()),
                'n_docs': group_mask.sum()
            }
        
        return {
            'topic_id': topic_id,
            'framings_by_group': framings
        }

# Made with Bob
