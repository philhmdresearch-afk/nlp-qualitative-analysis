"""
Text Preprocessing Module for Unstructured Data
Energy-efficient NLP preprocessing with NLTK
"""

import re
import string
from typing import List, Optional, Set, Dict
import pandas as pd
import numpy as np
from collections import Counter

# Lazy imports for energy efficiency
_nltk_loaded = False
_stopwords = None
_lemmatizer = None


def _ensure_nltk_resources():
    """Lazy load NLTK resources only when needed."""
    global _nltk_loaded, _stopwords, _lemmatizer
    
    if not _nltk_loaded:
        import nltk
        
        # Download required resources if not present
        required_resources = [
            'punkt_tab',  # New tokenizer format
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'  # Open Multilingual Wordnet
        ]
        
        for resource_name in required_resources:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                print(f"Note: Could not download {resource_name}: {e}")
        
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        _stopwords = set(stopwords.words('english'))
        _lemmatizer = WordNetLemmatizer()
        _nltk_loaded = True


class TextPreprocessor:
    """
    Efficient text preprocessing for unstructured data analysis.
    
    Features:
    - Text cleaning (URLs, emails, special characters)
    - Tokenization
    - Stopword removal
    - Lemmatization
    - N-gram extraction
    - Memory-efficient batch processing
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_length: int = 2,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        Initialize text preprocessor.
        
        Parameters:
        -----------
        lowercase : bool
            Convert text to lowercase
        remove_punctuation : bool
            Remove punctuation marks
        remove_numbers : bool
            Remove numeric characters
        remove_stopwords : bool
            Remove common stopwords
        lemmatize : bool
            Apply lemmatization
        min_word_length : int
            Minimum word length to keep
        custom_stopwords : set, optional
            Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.custom_stopwords = custom_stopwords or set()
        
        # Lazy load NLTK resources
        if self.remove_stopwords or self.lemmatize:
            _ensure_nltk_resources()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing URLs, emails, and special characters.
        
        Parameters:
        -----------
        text : str
            Raw text to clean
            
        Returns:
        --------
        str
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        text = self.email_pattern.sub(' ', text)
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Parameters:
        -----------
        text : str
            Text to tokenize
            
        Returns:
        --------
        list
            List of tokens
        """
        import nltk
        
        # Simple word tokenization
        tokens = nltk.word_tokenize(text)
        
        # Filter by length
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            stopwords_set = _stopwords.union(self.custom_stopwords)
            tokens = [t for t in tokens if t.lower() not in stopwords_set]
        
        # Lemmatize if specified
        if self.lemmatize and _lemmatizer is not None:
            try:
                tokens = [_lemmatizer.lemmatize(t) for t in tokens]
            except Exception as e:
                # Fallback: skip lemmatization if it fails
                print(f"Lemmatization skipped due to error: {e}")
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean and tokenize.
        
        Parameters:
        -----------
        text : str
            Raw text
            
        Returns:
        --------
        str
            Preprocessed text (space-separated tokens)
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)
    
    def preprocess_batch(
        self,
        texts: List[str],
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> List[str]:
        """
        Preprocess multiple texts efficiently in batches.
        
        Parameters:
        -----------
        texts : list
            List of raw texts
        batch_size : int
            Number of texts to process at once
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        list
            List of preprocessed texts
        """
        from tqdm import tqdm
        
        processed = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Preprocessing texts")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            processed.extend([self.preprocess(text) for text in batch])
        
        return processed
    
    def extract_ngrams(
        self,
        text: str,
        n: int = 2,
        min_freq: int = 1
    ) -> List[str]:
        """
        Extract n-grams from text.
        
        Parameters:
        -----------
        text : str
            Preprocessed text
        n : int
            N-gram size (2 for bigrams, 3 for trigrams, etc.)
        min_freq : int
            Minimum frequency to include n-gram
            
        Returns:
        --------
        list
            List of n-grams
        """
        tokens = text.split()
        
        if len(tokens) < n:
            return []
        
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Filter by frequency if specified
        if min_freq > 1:
            ngram_counts = Counter(ngrams)
            ngrams = [ng for ng in ngrams if ngram_counts[ng] >= min_freq]
        
        return ngrams
    
    def get_vocabulary(
        self,
        texts: List[str],
        max_features: Optional[int] = None,
        min_df: int = 1
    ) -> Dict[str, int]:
        """
        Build vocabulary from texts.
        
        Parameters:
        -----------
        texts : list
            List of preprocessed texts
        max_features : int, optional
            Maximum number of features to keep (most frequent)
        min_df : int
            Minimum document frequency
            
        Returns:
        --------
        dict
            Vocabulary mapping word to index
        """
        # Count word frequencies across documents
        word_doc_freq = Counter()
        
        for text in texts:
            unique_words = set(text.split())
            word_doc_freq.update(unique_words)
        
        # Filter by minimum document frequency
        vocab = {
            word: freq for word, freq in word_doc_freq.items()
            if freq >= min_df
        }
        
        # Sort by frequency and limit if specified
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        
        if max_features:
            sorted_vocab = sorted_vocab[:max_features]
        
        # Create word to index mapping
        vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
        
        return vocabulary
    
    def get_statistics(self, texts: List[str]) -> Dict:
        """
        Get preprocessing statistics.
        
        Parameters:
        -----------
        texts : list
            List of preprocessed texts
            
        Returns:
        --------
        dict
            Statistics about the texts
        """
        all_tokens = []
        doc_lengths = []
        
        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)
            doc_lengths.append(len(tokens))
        
        vocab = set(all_tokens)
        word_freq = Counter(all_tokens)
        
        return {
            'n_documents': len(texts),
            'vocabulary_size': len(vocab),
            'total_tokens': len(all_tokens),
            'avg_doc_length': np.mean(doc_lengths),
            'median_doc_length': np.median(doc_lengths),
            'min_doc_length': np.min(doc_lengths),
            'max_doc_length': np.max(doc_lengths),
            'top_10_words': word_freq.most_common(10)
        }


def create_default_preprocessor() -> TextPreprocessor:
    """Create a preprocessor with default settings for general use."""
    return TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
        remove_stopwords=True,
        lemmatize=True,
        min_word_length=2
    )


def create_minimal_preprocessor() -> TextPreprocessor:
    """Create a minimal preprocessor for faster processing."""
    return TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
        remove_stopwords=True,
        lemmatize=False,  # Skip lemmatization for speed
        min_word_length=2
    )

# Made with Bob
