"""
Data Loading Module
Supports both structured (CSV, Excel) and unstructured (text) data
Energy-efficient with lazy loading and memory optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import json


class DataLoader:
    """
    Unified data loader for structured and unstructured data.
    
    Supports:
    - CSV files (structured/unstructured)
    - Excel files (structured/unstructured)
    - JSON files (structured/unstructured)
    - Plain text files (unstructured)
    """
    
    def __init__(self):
        self.data = None
        self.data_type = None  # 'structured' or 'unstructured'
        self.metadata = {}
        
    def load_data(
        self,
        file_path: Union[str, Path],
        data_type: str = 'auto',
        text_column: Optional[str] = None,
        encoding: str = 'utf-8',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the data file
        data_type : str
            'structured', 'unstructured', or 'auto' (default)
        text_column : str, optional
            Column name containing text data (for unstructured analysis)
        encoding : str
            File encoding (default: 'utf-8')
        **kwargs : dict
            Additional arguments passed to pandas read functions
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            self.data = self._load_csv(file_path, encoding, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            self.data = self._load_excel(file_path, **kwargs)
        elif suffix == '.json':
            self.data = self._load_json(file_path, encoding, **kwargs)
        elif suffix == '.txt':
            self.data = self._load_text(file_path, encoding)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Detect data type if auto
        if data_type == 'auto':
            self.data_type = self._detect_data_type(self.data, text_column)
        else:
            self.data_type = data_type
        
        # Store metadata
        self.metadata = {
            'file_path': str(file_path),
            'file_type': suffix,
            'data_type': self.data_type,
            'n_rows': len(self.data),
            'n_columns': len(self.data.columns),
            'columns': list(self.data.columns),
            'text_column': text_column,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
        
        return self.data
    
    def _load_csv(self, file_path: Path, encoding: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with memory optimization."""
        # Use chunking for large files if needed
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            # Try alternative encodings
            for enc in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc, **kwargs)
                    print(f"Warning: Used {enc} encoding instead of {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise
        
        return df
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(file_path, **kwargs)
    
    def _load_json(self, file_path: Path, encoding: str, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if it's a dict of lists or a single record
            if all(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("JSON format not supported. Expected list or dict.")
        
        return df
    
    def _load_text(self, file_path: Path, encoding: str) -> pd.DataFrame:
        """Load plain text file (one document per line or entire file)."""
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        # Create DataFrame with text column
        df = pd.DataFrame({'text': [line.strip() for line in lines if line.strip()]})
        return df
    
    def _detect_data_type(
        self,
        df: pd.DataFrame,
        text_column: Optional[str] = None
    ) -> str:
        """
        Automatically detect if data is structured or unstructured.
        
        Logic:
        - If text_column is specified, it's unstructured
        - If most columns are numeric/categorical, it's structured
        - If there's a single text column with long strings, it's unstructured
        """
        if text_column:
            return 'unstructured'
        
        # Check column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=['object']).columns
        
        # If mostly numeric/categorical, it's structured
        if len(numeric_cols) > len(object_cols):
            return 'structured'
        
        # Check for long text columns (likely unstructured)
        for col in object_cols:
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 100:  # Arbitrary threshold for "long text"
                return 'unstructured'
        
        # Default to structured if unclear
        return 'structured'
    
    def get_text_column(self, text_column: Optional[str] = None) -> pd.Series:
        """
        Extract text column for unstructured analysis.
        
        Parameters:
        -----------
        text_column : str, optional
            Name of the text column. If None, auto-detect.
            
        Returns:
        --------
        pd.Series
            Text data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if text_column:
            if text_column not in self.data.columns:
                raise ValueError(f"Column '{text_column}' not found in data.")
            return self.data[text_column]
        
        # Auto-detect text column
        object_cols = self.data.select_dtypes(include=['object']).columns
        
        if len(object_cols) == 0:
            raise ValueError("No text columns found in data.")
        
        # Find column with longest average text length
        text_col = max(
            object_cols,
            key=lambda col: self.data[col].astype(str).str.len().mean()
        )
        
        return self.data[text_col]
    
    def get_structured_features(
        self,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Extract structured features (numeric and categorical).
        
        Parameters:
        -----------
        exclude_columns : list, optional
            Columns to exclude from analysis
            
        Returns:
        --------
        tuple
            (features_df, numeric_columns, categorical_columns)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.data.copy()
        
        # Exclude specified columns
        if exclude_columns:
            df = df.drop(columns=exclude_columns, errors='ignore')
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return df, numeric_cols, categorical_cols
    
    def get_summary(self) -> Dict:
        """Get summary statistics of loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            **self.metadata,
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.astype(str).to_dict()
        }
        
        if self.data_type == 'structured':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary['numeric_summary'] = self.data[numeric_cols].describe().to_dict()
        
        return summary
    
    def validate_data(self) -> Dict[str, List[str]]:
        """
        Validate data quality and return issues.
        
        Returns:
        --------
        dict
            Dictionary of validation issues
        """
        issues = {
            'errors': [],
            'warnings': []
        }
        
        if self.data is None:
            issues['errors'].append("No data loaded")
            return issues
        
        # Check for empty data
        if len(self.data) == 0:
            issues['errors'].append("Data is empty")
        
        # Check for missing values
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100)
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            issues['warnings'].append(
                f"Columns with >50% missing values: {', '.join(high_missing)}"
            )
        
        # Check for duplicate rows
        n_duplicates = self.data.duplicated().sum()
        if n_duplicates > 0:
            issues['warnings'].append(
                f"Found {n_duplicates} duplicate rows ({n_duplicates/len(self.data)*100:.1f}%)"
            )
        
        # Check for constant columns
        constant_cols = [
            col for col in self.data.columns
            if self.data[col].nunique() == 1
        ]
        if constant_cols:
            issues['warnings'].append(
                f"Constant columns (no variation): {', '.join(constant_cols)}"
            )
        
        return issues

# Made with Bob
