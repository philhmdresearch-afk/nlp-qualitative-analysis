"""
Sentiment Analysis Module using VADER
Rule-based sentiment analysis for text data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Lazy import for VADER
_vader_loaded = False
_vader_analyzer = None


def _ensure_vader():
    """Lazy load VADER sentiment analyzer."""
    global _vader_loaded, _vader_analyzer
    
    if not _vader_loaded:
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Download VADER lexicon if not present
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                print("Downloading VADER lexicon...")
                nltk.download('vader_lexicon', quiet=True)
            
            _vader_analyzer = SentimentIntensityAnalyzer()
            _vader_loaded = True
        except Exception as e:
            raise ImportError(f"Failed to load VADER: {e}")


class SentimentAnalyzer:
    """
    Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    VADER is a rule-based sentiment analysis tool that is:
    - Fast and efficient (no ML model needed)
    - Good for social media and short texts
    - Provides compound, positive, negative, and neutral scores
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        _ensure_vader()
        self.analyzer = _vader_analyzer
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Sentiment scores: {
                'compound': overall sentiment (-1 to 1),
                'pos': positive score (0 to 1),
                'neu': neutral score (0 to 1),
                'neg': negative score (0 to 1)
            }
        """
        if not isinstance(text, str):
            text = str(text)
        
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def analyze_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze sentiment for multiple texts.
        
        Parameters:
        -----------
        texts : list
            List of texts to analyze
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with sentiment scores for each text
        """
        from tqdm import tqdm
        
        results = []
        
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="Analyzing sentiment")
        
        for text in iterator:
            scores = self.analyze_text(text)
            results.append(scores)
        
        df = pd.DataFrame(results)
        
        # Add sentiment label based on compound score
        df['sentiment'] = df['compound'].apply(self._classify_sentiment)
        
        return df
    
    def _classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score.
        
        Thresholds:
        - Positive: compound >= 0.05
        - Negative: compound <= -0.05
        - Neutral: -0.05 < compound < 0.05
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def get_sentiment_distribution(
        self,
        texts: List[str]
    ) -> Dict[str, int]:
        """
        Get distribution of sentiment labels.
        
        Parameters:
        -----------
        texts : list
            List of texts
            
        Returns:
        --------
        dict
            Count of each sentiment label
        """
        sentiments = self.analyze_batch(texts, show_progress=False)
        return sentiments['sentiment'].value_counts().to_dict()
    
    def get_sentiment_by_group(
        self,
        texts: List[str],
        groups: List[str]
    ) -> pd.DataFrame:
        """
        Get sentiment statistics by group.
        
        Parameters:
        -----------
        texts : list
            List of texts
        groups : list
            Group labels for each text
            
        Returns:
        --------
        pd.DataFrame
            Sentiment statistics by group
        """
        sentiments = self.analyze_batch(texts, show_progress=False)
        sentiments['group'] = groups
        
        # Aggregate by group
        group_stats = sentiments.groupby('group').agg({
            'compound': ['mean', 'std', 'min', 'max'],
            'pos': 'mean',
            'neu': 'mean',
            'neg': 'mean',
            'sentiment': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        return group_stats
    
    def get_most_positive_negative(
        self,
        texts: List[str],
        n: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most positive and negative texts.
        
        Parameters:
        -----------
        texts : list
            List of texts
        n : int
            Number of texts to return for each category
            
        Returns:
        --------
        dict
            Dictionary with 'most_positive' and 'most_negative' texts
        """
        sentiments = self.analyze_batch(texts, show_progress=False)
        sentiments['text'] = texts
        
        # Sort by compound score
        sorted_df = sentiments.sort_values('compound', ascending=False)
        
        most_positive = [
            (row['text'], row['compound'])
            for _, row in sorted_df.head(n).iterrows()
        ]
        
        most_negative = [
            (row['text'], row['compound'])
            for _, row in sorted_df.tail(n).iterrows()
        ]
        
        return {
            'most_positive': most_positive,
            'most_negative': most_negative
        }
    
    def get_statistics(self, texts: List[str]) -> Dict:
        """
        Get overall sentiment statistics.
        
        Parameters:
        -----------
        texts : list
            List of texts
            
        Returns:
        --------
        dict
            Sentiment statistics
        """
        sentiments = self.analyze_batch(texts, show_progress=False)
        
        return {
            'n_texts': len(texts),
            'avg_compound': sentiments['compound'].mean(),
            'std_compound': sentiments['compound'].std(),
            'avg_positive': sentiments['pos'].mean(),
            'avg_neutral': sentiments['neu'].mean(),
            'avg_negative': sentiments['neg'].mean(),
            'sentiment_distribution': sentiments['sentiment'].value_counts().to_dict(),
            'positive_pct': (sentiments['sentiment'] == 'positive').sum() / len(texts) * 100,
            'neutral_pct': (sentiments['sentiment'] == 'neutral').sum() / len(texts) * 100,
            'negative_pct': (sentiments['sentiment'] == 'negative').sum() / len(texts) * 100
        }

# Made with Bob
