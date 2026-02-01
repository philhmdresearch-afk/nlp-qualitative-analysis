"""
Quote Analysis Module for UX Research
Provenance tracking, representativeness, and traceability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


class QuoteAnalyzer:
    """
    Quote analysis for qualitative research traceability.
    
    Features:
    - Quote provenance chains
    - Representativeness indicators
    - Exemplar quote selection
    - Participant contribution tracking
    """
    
    def __init__(self):
        """Initialize quote analyzer."""
        self.quote_metadata = {}
        
    def create_provenance_chain(
        self,
        texts: List[str],
        document_topic_matrix: np.ndarray,
        topic_id: int,
        metadata: Optional[pd.DataFrame] = None,
        n_quotes: int = 5
    ) -> Dict:
        """
        Create provenance chain for a theme with supporting quotes.
        
        Parameters:
        -----------
        texts : List[str]
            Original texts
        document_topic_matrix : np.ndarray
            Document-topic distribution
        topic_id : int
            Topic to analyze
        metadata : pd.DataFrame, optional
            Additional metadata (participant_id, session, timestamp, etc.)
        n_quotes : int
            Number of quotes to include
            
        Returns:
        --------
        Dict with provenance information
        """
        topic_probs = document_topic_matrix[:, topic_id]
        
        # Get top documents for this topic
        top_indices = topic_probs.argsort()[-n_quotes:][::-1]
        
        quotes = []
        contributors = set()
        sessions = set()
        
        for idx in top_indices:
            quote_data = {
                'text': texts[idx],
                'topic_strength': float(topic_probs[idx]),
                'document_id': int(idx)
            }
            
            # Add metadata if available
            if metadata is not None and idx < len(metadata):
                row = metadata.iloc[idx]
                
                if 'participant_id' in row:
                    quote_data['participant_id'] = row['participant_id']
                    contributors.add(row['participant_id'])
                
                if 'session' in row:
                    quote_data['session'] = row['session']
                    sessions.add(row['session'])
                
                if 'timestamp' in row:
                    quote_data['timestamp'] = row['timestamp']
                
                if 'task' in row:
                    quote_data['task'] = row['task']
                
                if 'context' in row:
                    quote_data['context'] = row['context']
            
            quotes.append(quote_data)
        
        # Compute diversity metrics
        n_contributors = len(contributors)
        n_sessions = len(sessions)
        n_total_docs = (topic_probs > 0.05).sum()
        
        return {
            'topic_id': topic_id,
            'quotes': quotes,
            'n_contributors': n_contributors,
            'n_sessions': n_sessions,
            'n_supporting_documents': int(n_total_docs),
            'contributors': list(contributors),
            'sessions': list(sessions),
            'coverage': f"{n_contributors} participants across {n_sessions} sessions"
        }
    
    def assess_quote_representativeness(
        self,
        quote_text: str,
        all_texts: List[str],
        document_topic_matrix: np.ndarray,
        topic_id: int,
        vectorizer
    ) -> Dict:
        """
        Assess how representative a quote is of its theme.
        
        Parameters:
        -----------
        quote_text : str
            The quote to assess
        all_texts : List[str]
            All texts in the corpus
        document_topic_matrix : np.ndarray
            Document-topic distribution
        topic_id : int
            Topic this quote represents
        vectorizer : fitted vectorizer
            TF-IDF or Count vectorizer
            
        Returns:
        --------
        Dict with representativeness metrics
        """
        # Find the quote in the corpus
        try:
            quote_idx = all_texts.index(quote_text)
        except ValueError:
            # Quote not found exactly, find closest match
            quote_vec = vectorizer.transform([quote_text])
            all_vecs = vectorizer.transform(all_texts)
            similarities = cosine_similarity(quote_vec, all_vecs)[0]
            quote_idx = similarities.argmax()
        
        # Get documents strongly associated with this topic
        topic_probs = document_topic_matrix[:, topic_id]
        topic_docs = topic_probs > 0.1
        
        if topic_docs.sum() == 0:
            return {
                'representativeness': 'Unknown',
                'reason': 'No documents strongly associated with this topic'
            }
        
        # Compute quote's topic strength
        quote_topic_strength = topic_probs[quote_idx]
        
        # Compare to other documents in this topic
        topic_strengths = topic_probs[topic_docs]
        percentile = (topic_strengths < quote_topic_strength).sum() / len(topic_strengths) * 100
        
        # Compute semantic similarity to other topic documents
        quote_vec = vectorizer.transform([all_texts[quote_idx]])
        topic_texts = [all_texts[i] for i, is_topic in enumerate(topic_docs) if is_topic]
        topic_vecs = vectorizer.transform(topic_texts)
        
        similarities = cosine_similarity(quote_vec, topic_vecs)[0]
        avg_similarity = similarities.mean()
        
        # Determine representativeness
        if percentile >= 75 and avg_similarity >= 0.3:
            label = "Highly Representative"
            reason = f"Strong topic association (top {100-percentile:.0f}%) and high semantic similarity"
        elif percentile >= 50 and avg_similarity >= 0.2:
            label = "Representative"
            reason = "Moderate topic association and semantic similarity"
        elif percentile >= 25:
            label = "Illustrative but Uncommon"
            reason = "Lower topic association, may represent edge case"
        else:
            label = "Edge Case"
            reason = "Weak topic association, not typical of this theme"
        
        return {
            'representativeness': label,
            'topic_strength': float(quote_topic_strength),
            'percentile': float(percentile),
            'avg_similarity': float(avg_similarity),
            'reason': reason,
            'confidence': 'High' if percentile >= 60 else 'Medium' if percentile >= 30 else 'Low'
        }
    
    def select_exemplar_quotes(
        self,
        texts: List[str],
        document_topic_matrix: np.ndarray,
        topic_id: int,
        vectorizer,
        n_quotes: int = 3,
        diversity_weight: float = 0.3
    ) -> List[Dict]:
        """
        Select diverse, representative exemplar quotes for a theme.
        
        Parameters:
        -----------
        texts : List[str]
            All texts
        document_topic_matrix : np.ndarray
            Document-topic distribution
        topic_id : int
            Topic to select quotes for
        vectorizer : fitted vectorizer
            For computing semantic similarity
        n_quotes : int
            Number of quotes to select
        diversity_weight : float
            Weight for diversity vs representativeness (0-1)
            
        Returns:
        --------
        List of exemplar quotes with metadata
        """
        topic_probs = document_topic_matrix[:, topic_id]
        
        # Get candidate documents (strong topic association)
        candidates = topic_probs > 0.1
        candidate_indices = np.where(candidates)[0]
        
        if len(candidate_indices) == 0:
            return []
        
        # Vectorize candidates
        candidate_texts = [texts[i] for i in candidate_indices]
        candidate_vecs = vectorizer.transform(candidate_texts)
        
        # Select quotes using MMR (Maximal Marginal Relevance)
        selected_indices = []
        selected_vecs = []
        
        for _ in range(min(n_quotes, len(candidate_indices))):
            if not selected_indices:
                # First quote: highest topic probability
                best_idx = topic_probs[candidate_indices].argmax()
                selected_indices.append(candidate_indices[best_idx])
                selected_vecs.append(candidate_vecs[best_idx])
            else:
                # Subsequent quotes: balance relevance and diversity
                scores = []
                
                for i, idx in enumerate(candidate_indices):
                    if idx in selected_indices:
                        scores.append(-np.inf)
                        continue
                    
                    # Relevance: topic probability
                    relevance = topic_probs[idx]
                    
                    # Diversity: minimum similarity to selected quotes
                    similarities = cosine_similarity(
                        candidate_vecs[i].reshape(1, -1),
                        np.vstack(selected_vecs)
                    )[0]
                    diversity = 1 - similarities.max()
                    
                    # Combined score
                    score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                    scores.append(score)
                
                best_idx = np.argmax(scores)
                selected_indices.append(candidate_indices[best_idx])
                selected_vecs.append(candidate_vecs[best_idx])
        
        # Create exemplar quote objects
        exemplars = []
        for idx in selected_indices:
            representativeness = self.assess_quote_representativeness(
                texts[idx], texts, document_topic_matrix, topic_id, vectorizer
            )
            
            exemplars.append({
                'text': texts[idx],
                'document_id': int(idx),
                'topic_strength': float(topic_probs[idx]),
                'representativeness': representativeness['representativeness'],
                'confidence': representativeness['confidence']
            })
        
        return exemplars
    
    def track_participant_contributions(
        self,
        texts: List[str],
        metadata: pd.DataFrame,
        document_topic_matrix: np.ndarray,
        sentiment_scores: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Track participant contributions and identify imbalances.
        
        Parameters:
        -----------
        texts : List[str]
            All texts
        metadata : pd.DataFrame
            Must include 'participant_id' column
        document_topic_matrix : np.ndarray
            Document-topic distribution
        sentiment_scores : np.ndarray, optional
            Sentiment scores per document
            
        Returns:
        --------
        Dict with participation metrics and warnings
        """
        if 'participant_id' not in metadata.columns:
            return {'error': 'participant_id column required in metadata'}
        
        participant_ids = metadata['participant_id'].values
        unique_participants = set(participant_ids)
        
        # Count contributions per participant
        contribution_counts = Counter(participant_ids)
        
        # Compute participation metrics
        total_docs = len(texts)
        n_participants = len(unique_participants)
        avg_contributions = total_docs / n_participants
        
        # Identify dominant participants
        dominant_threshold = avg_contributions * 2
        dominant_participants = {
            pid: count for pid, count in contribution_counts.items()
            if count > dominant_threshold
        }
        
        # Compute sentiment contribution if available
        sentiment_warnings = []
        if sentiment_scores is not None:
            for pid in unique_participants:
                pid_mask = participant_ids == pid
                pid_sentiments = sentiment_scores[pid_mask]
                
                # Check if this participant drives negative sentiment
                if pid_sentiments.mean() < -0.2 and pid_mask.sum() > avg_contributions:
                    pct_negative = (pid_sentiments < -0.05).sum() / len(pid_sentiments) * 100
                    sentiment_warnings.append({
                        'participant_id': pid,
                        'avg_sentiment': float(pid_sentiments.mean()),
                        'n_contributions': int(pid_mask.sum()),
                        'pct_negative': float(pct_negative),
                        'warning': f"Participant {pid} contributes {pct_negative:.0f}% negative sentiment"
                    })
        
        # Topic concentration warnings
        topic_warnings = []
        n_topics = document_topic_matrix.shape[1]
        
        for topic_id in range(n_topics):
            topic_probs = document_topic_matrix[:, topic_id]
            topic_docs = topic_probs > 0.1
            
            if topic_docs.sum() > 0:
                # Check participant concentration in this topic
                topic_participants = participant_ids[topic_docs]
                topic_contribution_counts = Counter(topic_participants)
                
                # Find if one participant dominates this topic
                max_contributor = topic_contribution_counts.most_common(1)[0]
                contribution_pct = max_contributor[1] / len(topic_participants) * 100
                
                if contribution_pct > 40:  # One participant > 40% of topic
                    topic_warnings.append({
                        'topic_id': topic_id,
                        'dominant_participant': max_contributor[0],
                        'contribution_pct': float(contribution_pct),
                        'warning': f"Topic {topic_id}: {contribution_pct:.0f}% from participant {max_contributor[0]}"
                    })
        
        return {
            'n_participants': n_participants,
            'total_contributions': total_docs,
            'avg_contributions_per_participant': avg_contributions,
            'contribution_distribution': dict(contribution_counts),
            'dominant_participants': dominant_participants,
            'participation_imbalance': len(dominant_participants) > 0,
            'sentiment_warnings': sentiment_warnings,
            'topic_concentration_warnings': topic_warnings,
            'gini_coefficient': self._compute_gini(list(contribution_counts.values()))
        }
    
    def _compute_gini(self, values: List[int]) -> float:
        """
        Compute Gini coefficient for participation inequality.
        0 = perfect equality, 1 = perfect inequality
        """
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return float((2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n)
    
    def detect_question_bias(
        self,
        texts: List[str],
        metadata: pd.DataFrame,
        document_topic_matrix: np.ndarray,
        sentiment_scores: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Detect if certain questions drive specific themes or sentiment.
        
        Parameters:
        -----------
        texts : List[str]
            All texts
        metadata : pd.DataFrame
            Must include 'question' or 'prompt' column
        document_topic_matrix : np.ndarray
            Document-topic distribution
        sentiment_scores : np.ndarray, optional
            Sentiment scores
            
        Returns:
        --------
        Dict with question bias analysis
        """
        question_col = None
        for col in ['question', 'prompt', 'question_id']:
            if col in metadata.columns:
                question_col = col
                break
        
        if question_col is None:
            return {'error': 'No question/prompt column found in metadata'}
        
        questions = metadata[question_col].values
        unique_questions = set(questions)
        
        question_analysis = {}
        
        for question in unique_questions:
            q_mask = questions == question
            q_docs = document_topic_matrix[q_mask]
            
            # Compute topic distribution for this question
            q_topic_dist = q_docs.mean(axis=0)
            
            # Find dominant topics for this question
            dominant_topics = np.where(q_topic_dist > 0.15)[0]
            
            analysis = {
                'n_responses': int(q_mask.sum()),
                'dominant_topics': dominant_topics.tolist(),
                'topic_distribution': q_topic_dist.tolist()
            }
            
            # Sentiment analysis if available
            if sentiment_scores is not None:
                q_sentiments = sentiment_scores[q_mask]
                analysis['avg_sentiment'] = float(q_sentiments.mean())
                analysis['sentiment_std'] = float(q_sentiments.std())
                
                # Check if this question drives negative sentiment
                if q_sentiments.mean() < -0.1:
                    analysis['bias_warning'] = "This question may elicit negative responses"
                elif q_sentiments.mean() > 0.1:
                    analysis['bias_warning'] = "This question may elicit positive responses"
            
            question_analysis[str(question)] = analysis
        
        return {
            'n_questions': len(unique_questions),
            'question_analysis': question_analysis,
            'recommendations': self._generate_question_recommendations(question_analysis)
        }
    
    def _generate_question_recommendations(self, question_analysis: Dict) -> List[str]:
        """Generate recommendations based on question bias analysis."""
        recommendations = []
        
        for question, analysis in question_analysis.items():
            if 'bias_warning' in analysis:
                recommendations.append(
                    f"Consider rephrasing: '{question[:50]}...' - {analysis['bias_warning']}"
                )
            
            if len(analysis.get('dominant_topics', [])) == 1:
                recommendations.append(
                    f"Question '{question[:50]}...' only elicits one theme - consider broadening"
                )
        
        return recommendations

# Made with Bob
