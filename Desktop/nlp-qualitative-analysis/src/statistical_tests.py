"""
Statistical Testing Module
Chi-square tests and other statistical analyses for cluster/topic distributions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import mannwhitneyu, kruskal


def cramers_v(contingency_table: np.ndarray) -> float:
    """
    Calculate Cramér's V statistic for categorical association.
    
    Args:
        contingency_table: 2D array of observed frequencies
        
    Returns:
        Cramér's V value (0 to 1)
    """
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
import warnings

warnings.filterwarnings('ignore')


class StatisticalTester:
    """
    Statistical tests for comparing distributions across groups.
    
    Features:
    - Chi-square test for independence
    - Fisher's exact test (for small samples)
    - Cramér's V (effect size)
    - Mann-Whitney U test (non-parametric)
    - Kruskal-Wallis test (multiple groups)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical tester.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
        """
        self.alpha = alpha
        
    def chi_square_test(
        self,
        contingency_table: np.ndarray,
        correction: bool = True
    ) -> Dict:
        """
        Perform chi-square test of independence.
        
        Parameters:
        -----------
        contingency_table : np.ndarray
            Contingency table (rows x columns)
        correction : bool
            Apply Yates' correction for continuity
            
        Returns:
        --------
        dict
            Test results including statistic, p-value, effect size
        """
        chi2, p_value, dof, expected = chi2_contingency(
            contingency_table,
            correction=correction
        )
        
        # Compute Cramér's V (effect size)
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'test': 'chi_square',
            'statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'significant': p_value < self.alpha,
            'expected_frequencies': expected
        }
    
    def fisher_exact_test(
        self,
        contingency_table: np.ndarray
    ) -> Dict:
        """
        Perform Fisher's exact test (for 2x2 tables).
        
        Parameters:
        -----------
        contingency_table : np.ndarray
            2x2 contingency table
            
        Returns:
        --------
        dict
            Test results
        """
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 contingency table")
        
        odds_ratio, p_value = fisher_exact(contingency_table)
        
        return {
            'test': 'fisher_exact',
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def compare_topic_distributions(
        self,
        group1_topics: List[int],
        group2_topics: List[int],
        topic_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare topic distributions between two groups.
        
        Parameters:
        -----------
        group1_topics : list
            Topic assignments for group 1
        group2_topics : list
            Topic assignments for group 2
        topic_labels : list, optional
            Labels for topics
            
        Returns:
        --------
        dict
            Comparison results with chi-square test
        """
        # Create contingency table
        all_topics = sorted(set(group1_topics + group2_topics))
        
        if topic_labels is None:
            topic_labels = [f"Topic {i}" for i in all_topics]
        
        contingency = []
        for topic in all_topics:
            row = [
                group1_topics.count(topic),
                group2_topics.count(topic)
            ]
            contingency.append(row)
        
        contingency_table = np.array(contingency)
        
        # Perform chi-square test
        test_results = self.chi_square_test(contingency_table)
        
        # Add topic-specific information
        test_results['topics'] = topic_labels
        test_results['group1_counts'] = contingency_table[:, 0].tolist()
        test_results['group2_counts'] = contingency_table[:, 1].tolist()
        test_results['group1_proportions'] = (
            contingency_table[:, 0] / contingency_table[:, 0].sum()
        ).tolist()
        test_results['group2_proportions'] = (
            contingency_table[:, 1] / contingency_table[:, 1].sum()
        ).tolist()
        
        return test_results
    
    def compare_cluster_distributions(
        self,
        labels: np.ndarray,
        groups: np.ndarray
    ) -> Dict:
        """
        Compare cluster distributions across groups.
        
        Parameters:
        -----------
        labels : np.ndarray
            Cluster labels
        groups : np.ndarray
            Group assignments
            
        Returns:
        --------
        dict
            Comparison results
        """
        # Create contingency table
        unique_clusters = np.unique(labels)
        unique_groups = np.unique(groups)
        
        contingency = np.zeros((len(unique_clusters), len(unique_groups)))
        
        for i, cluster in enumerate(unique_clusters):
            for j, group in enumerate(unique_groups):
                mask = (labels == cluster) & (groups == group)
                contingency[i, j] = mask.sum()
        
        # Perform chi-square test
        test_results = self.chi_square_test(contingency)
        
        # Add cluster and group information
        test_results['clusters'] = unique_clusters.tolist()
        test_results['groups'] = unique_groups.tolist()
        test_results['contingency_table'] = contingency.tolist()
        
        return test_results
    
    def pairwise_comparisons(
        self,
        data: Dict[str, List[int]],
        correction: str = 'bonferroni'
    ) -> pd.DataFrame:
        """
        Perform pairwise chi-square tests with multiple comparison correction.
        
        Parameters:
        -----------
        data : dict
            Dictionary mapping group names to topic/cluster assignments
        correction : str
            Multiple comparison correction: 'bonferroni', 'fdr', or 'none'
            
        Returns:
        --------
        pd.DataFrame
            Pairwise comparison results
        """
        from itertools import combinations
        
        group_names = list(data.keys())
        results = []
        
        # Perform all pairwise comparisons
        for group1, group2 in combinations(group_names, 2):
            comparison = self.compare_topic_distributions(
                data[group1],
                data[group2]
            )
            
            results.append({
                'group1': group1,
                'group2': group2,
                'chi2': comparison['statistic'],
                'p_value': comparison['p_value'],
                'cramers_v': comparison['cramers_v'],
                'significant': comparison['significant']
            })
        
        df = pd.DataFrame(results)
        
        # Apply multiple comparison correction
        if correction == 'bonferroni':
            n_comparisons = len(df)
            df['p_value_corrected'] = df['p_value'] * n_comparisons
            df['p_value_corrected'] = df['p_value_corrected'].clip(upper=1.0)
            df['significant_corrected'] = df['p_value_corrected'] < self.alpha
        elif correction == 'fdr':
            # Benjamini-Hochberg FDR correction
            df = df.sort_values('p_value')
            n = len(df)
            df['rank'] = range(1, n + 1)
            df['p_value_corrected'] = df['p_value'] * n / df['rank']
            df['p_value_corrected'] = df['p_value_corrected'].clip(upper=1.0)
            df['significant_corrected'] = df['p_value_corrected'] < self.alpha
        
        return df
    
    def mann_whitney_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Parameters:
        -----------
        group1 : np.ndarray
            Data for group 1
        group2 : np.ndarray
            Data for group 2
            
        Returns:
        --------
        dict
            Test results
        """
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        return {
            'test': 'mann_whitney',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'group1_median': np.median(group1),
            'group2_median': np.median(group2)
        }
    
    def kruskal_wallis_test(
        self,
        *groups: np.ndarray
    ) -> Dict:
        """
        Perform Kruskal-Wallis test (non-parametric ANOVA).
        
        Parameters:
        -----------
        *groups : np.ndarray
            Data for each group
            
        Returns:
        --------
        dict
            Test results
        """
        statistic, p_value = kruskal(*groups)
        
        return {
            'test': 'kruskal_wallis',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_groups': len(groups),
            'group_medians': [np.median(g) for g in groups]
        }


def create_contingency_table(
    labels1: np.ndarray,
    labels2: np.ndarray
) -> pd.DataFrame:
    """
    Create a contingency table from two sets of labels.
    
    Parameters:
    -----------
    labels1 : np.ndarray
        First set of labels (e.g., clusters)
    labels2 : np.ndarray
        Second set of labels (e.g., groups)
        
    Returns:
    --------
    pd.DataFrame
        Contingency table
    """
    return pd.crosstab(labels1, labels2, margins=True)


def compute_effect_size(contingency_table: np.ndarray) -> float:
    """
    Compute Cramér's V effect size.
    
    Parameters:
    -----------
    contingency_table : np.ndarray
        Contingency table
        
    Returns:
    --------
    float
        Cramér's V (0 to 1)
    """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    
    if min_dim == 0:
        return 0.0
    
    return np.sqrt(chi2 / (n * min_dim))

# Made with Bob
