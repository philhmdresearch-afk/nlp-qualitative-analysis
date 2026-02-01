"""
Structured Data Preprocessing Module
Handles numeric and categorical data with energy-efficient processing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class StructuredPreprocessor:
    """
    Preprocessing for structured (numeric/categorical) data.
    
    Features:
    - Automatic type detection
    - Missing value imputation
    - Numeric scaling (Standard, MinMax, Robust)
    - Categorical encoding (OneHot, Label, Ordinal)
    - Gower distance support for mixed types
    - Memory-efficient sparse matrix operations
    """
    
    def __init__(
        self,
        numeric_strategy: str = 'standard',
        categorical_strategy: str = 'onehot',
        handle_missing: str = 'mean',
        max_categories: int = 50
    ):
        """
        Initialize structured data preprocessor.
        
        Parameters:
        -----------
        numeric_strategy : str
            Scaling strategy: 'standard', 'minmax', 'robust', or 'none'
        categorical_strategy : str
            Encoding strategy: 'onehot', 'label', 'ordinal', or 'none'
        handle_missing : str
            Missing value strategy: 'mean', 'median', 'most_frequent', 'drop'
        max_categories : int
            Maximum unique categories for one-hot encoding
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.handle_missing = handle_missing
        self.max_categories = max_categories
        
        # Initialize transformers
        self.numeric_scaler = None
        self.categorical_encoder = None
        self.numeric_imputer = None
        self.categorical_imputer = None
        
        # Store column information
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_names = []
        self.is_fitted = False
        
    def fit(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> 'StructuredPreprocessor':
        """
        Fit preprocessor on data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        numeric_cols : list, optional
            Numeric column names (auto-detect if None)
        categorical_cols : list, optional
            Categorical column names (auto-detect if None)
            
        Returns:
        --------
        self
        """
        # Auto-detect column types if not provided
        if numeric_cols is None:
            self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_columns = numeric_cols
        
        if categorical_cols is None:
            self.categorical_columns = df.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        else:
            self.categorical_columns = categorical_cols
        
        # Fit numeric transformers
        if self.numeric_columns:
            self._fit_numeric(df[self.numeric_columns])
        
        # Fit categorical transformers
        if self.categorical_columns:
            self._fit_categorical(df[self.categorical_columns])
        
        self.is_fitted = True
        return self
    
    def _fit_numeric(self, df: pd.DataFrame):
        """Fit numeric transformers."""
        # Imputer
        if self.handle_missing != 'drop':
            strategy = self.handle_missing if self.handle_missing in ['mean', 'median'] else 'mean'
            self.numeric_imputer = SimpleImputer(strategy=strategy)
            self.numeric_imputer.fit(df)
        
        # Scaler
        if self.numeric_strategy == 'standard':
            self.numeric_scaler = StandardScaler()
        elif self.numeric_strategy == 'minmax':
            self.numeric_scaler = MinMaxScaler()
        elif self.numeric_strategy == 'robust':
            self.numeric_scaler = RobustScaler()
        
        if self.numeric_scaler:
            if self.numeric_imputer:
                imputed = self.numeric_imputer.transform(df)
                self.numeric_scaler.fit(imputed)
            else:
                self.numeric_scaler.fit(df)
    
    def _fit_categorical(self, df: pd.DataFrame):
        """Fit categorical transformers."""
        # Imputer
        if self.handle_missing != 'drop':
            self.categorical_imputer = SimpleImputer(
                strategy='most_frequent',
                fill_value='missing'
            )
            self.categorical_imputer.fit(df)
        
        # Check category counts
        high_cardinality_cols = []
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique > self.max_categories:
                high_cardinality_cols.append(col)
                print(f"Warning: Column '{col}' has {n_unique} unique values. "
                      f"Consider using 'label' encoding or reducing categories.")
        
        # Encoder
        if self.categorical_strategy == 'onehot':
            self.categorical_encoder = OneHotEncoder(
                sparse_output=True,  # Use sparse matrices for memory efficiency
                handle_unknown='ignore',
                max_categories=self.max_categories
            )
        elif self.categorical_strategy == 'label':
            # Use dictionary of label encoders for each column
            self.categorical_encoder = {}
            for col in df.columns:
                le = LabelEncoder()
                if self.categorical_imputer:
                    imputed = self.categorical_imputer.transform(df[[col]])
                    le.fit(imputed.ravel())
                else:
                    le.fit(df[col])
                self.categorical_encoder[col] = le
        elif self.categorical_strategy == 'ordinal':
            self.categorical_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
        
        if self.categorical_encoder and self.categorical_strategy != 'label':
            if self.categorical_imputer:
                imputed = self.categorical_imputer.transform(df)
                self.categorical_encoder.fit(imputed)
            else:
                self.categorical_encoder.fit(df)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
            
        Returns:
        --------
        np.ndarray
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        features = []
        
        # Transform numeric features
        if self.numeric_columns:
            numeric_data = df[self.numeric_columns].copy()
            
            if self.numeric_imputer:
                numeric_data = self.numeric_imputer.transform(numeric_data)
            
            if self.numeric_scaler:
                numeric_data = self.numeric_scaler.transform(numeric_data)
            
            features.append(numeric_data)
        
        # Transform categorical features
        if self.categorical_columns:
            categorical_data = df[self.categorical_columns].copy()
            
            if self.categorical_imputer:
                categorical_data = self.categorical_imputer.transform(categorical_data)
            
            if self.categorical_strategy == 'label':
                # Label encoding
                encoded = np.zeros((len(categorical_data), len(self.categorical_columns)))
                for i, col in enumerate(self.categorical_columns):
                    encoded[:, i] = self.categorical_encoder[col].transform(
                        categorical_data[:, i]
                    )
                features.append(encoded)
            elif self.categorical_encoder:
                # OneHot or Ordinal encoding
                encoded = self.categorical_encoder.transform(categorical_data)
                if hasattr(encoded, 'toarray'):
                    # Convert sparse to dense if needed
                    encoded = encoded.toarray()
                features.append(encoded)
        
        # Concatenate all features
        if len(features) == 0:
            raise ValueError("No features to transform")
        
        return np.hstack(features)
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df, numeric_cols, categorical_cols)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get names of transformed features."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        feature_names = []
        
        # Numeric feature names
        feature_names.extend(self.numeric_columns)
        
        # Categorical feature names
        if self.categorical_strategy == 'onehot' and self.categorical_encoder:
            cat_names = self.categorical_encoder.get_feature_names_out(
                self.categorical_columns
            )
            feature_names.extend(cat_names)
        elif self.categorical_strategy in ['label', 'ordinal']:
            feature_names.extend(self.categorical_columns)
        
        return feature_names
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get preprocessing statistics."""
        stats = {
            'n_samples': len(df),
            'n_numeric_features': len(self.numeric_columns),
            'n_categorical_features': len(self.categorical_columns),
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns
        }
        
        # Missing value statistics
        if self.numeric_columns:
            stats['numeric_missing'] = df[self.numeric_columns].isnull().sum().to_dict()
        
        if self.categorical_columns:
            stats['categorical_missing'] = df[self.categorical_columns].isnull().sum().to_dict()
            stats['categorical_cardinality'] = {
                col: df[col].nunique() for col in self.categorical_columns
            }
        
        return stats


def compute_gower_distance(
    X: np.ndarray,
    numeric_indices: List[int],
    categorical_indices: List[int],
    ranges: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute Gower distance for mixed numeric/categorical data.
    
    Gower distance handles mixed data types by computing:
    - Manhattan distance (normalized) for numeric features
    - Simple matching for categorical features
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    numeric_indices : list
        Indices of numeric columns
    categorical_indices : list
        Indices of categorical columns
    ranges : np.ndarray, optional
        Ranges for numeric features (for normalization)
        
    Returns:
    --------
    np.ndarray
        Distance matrix
    """
    try:
        import gower
        # Use gower library if available (more efficient)
        return gower.gower_matrix(X)
    except ImportError:
        # Fallback to manual implementation
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        # Compute ranges for numeric features if not provided
        if ranges is None and numeric_indices:
            ranges = np.ptp(X[:, numeric_indices], axis=0)
            ranges[ranges == 0] = 1  # Avoid division by zero
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = 0
                n_features = 0
                
                # Numeric features (Manhattan distance, normalized)
                if numeric_indices:
                    numeric_dist = np.abs(
                        X[i, numeric_indices] - X[j, numeric_indices]
                    ) / ranges
                    dist += np.sum(numeric_dist)
                    n_features += len(numeric_indices)
                
                # Categorical features (simple matching)
                if categorical_indices:
                    cat_dist = np.sum(
                        X[i, categorical_indices] != X[j, categorical_indices]
                    )
                    dist += cat_dist
                    n_features += len(categorical_indices)
                
                # Average distance
                distances[i, j] = distances[j, i] = dist / n_features if n_features > 0 else 0
        
        return distances

# Made with Bob
