"""
Visualization Module
Comprehensive visualizations for both structured and unstructured data analysis
Includes bar charts, dendrograms, word clouds, and more
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Comprehensive visualization toolkit for data analysis.
    
    Features:
    - Bar charts for distributions
    - Dendrograms for hierarchical clustering
    - Word clouds for text analysis
    - Scatter plots for clusters
    - Heatmaps for correlations
    - Interactive plots with Plotly
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        style : str
            Matplotlib style
        figsize : tuple
            Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
    def plot_distribution_bar(
        self,
        data: Union[pd.Series, Dict, List],
        title: str = "Distribution",
        xlabel: str = "Category",
        ylabel: str = "Count",
        color: str = 'steelblue',
        horizontal: bool = False,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart for distribution.
        
        Parameters:
        -----------
        data : pd.Series, dict, or list
            Data to plot
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        color : str
            Bar color
        horizontal : bool
            Create horizontal bar chart
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert data to appropriate format
        if isinstance(data, pd.Series):
            counts = data.value_counts()
        elif isinstance(data, dict):
            counts = pd.Series(data)
        elif isinstance(data, list):
            counts = pd.Series(data).value_counts()
        else:
            raise ValueError("Data must be pd.Series, dict, or list")
        
        # Create bar chart
        if horizontal:
            counts.plot(kind='barh', ax=ax, color=color)
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        else:
            counts.plot(kind='bar', ax=ax, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_grouped_bar(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: str,
        title: str = "Grouped Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create grouped bar chart.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to plot
        x_col : str
            Column for x-axis
        y_col : str
            Column for y-axis (values)
        hue_col : str
            Column for grouping
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dendrogram(
        self,
        linkage_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Hierarchical Clustering Dendrogram",
        orientation: str = 'top',
        truncate_mode: Optional[str] = None,
        p: int = 30,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create dendrogram for hierarchical clustering.
        
        Parameters:
        -----------
        linkage_matrix : np.ndarray
            Linkage matrix from hierarchical clustering
        labels : list, optional
            Sample labels
        title : str
            Plot title
        orientation : str
            Dendrogram orientation: 'top', 'bottom', 'left', 'right'
        truncate_mode : str, optional
            Truncation mode: 'lastp', 'level', None
        p : int
            Truncation parameter
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        from scipy.cluster.hierarchy import dendrogram
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dend = dendrogram(
            linkage_matrix,
            labels=labels,
            orientation=orientation,
            truncate_mode=truncate_mode,
            p=p,
            ax=ax,
            color_threshold=0.7 * max(linkage_matrix[:, 2])
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if orientation in ['top', 'bottom']:
            ax.set_xlabel('Sample Index' if labels is None else 'Samples')
            ax.set_ylabel('Distance')
            plt.xticks(rotation=90)
        else:
            ax.set_ylabel('Sample Index' if labels is None else 'Samples')
            ax.set_xlabel('Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dendrogram_interactive(
        self,
        linkage_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Interactive Dendrogram"
    ):
        """
        Create interactive dendrogram using Plotly.
        
        Parameters:
        -----------
        linkage_matrix : np.ndarray
            Linkage matrix
        labels : list, optional
            Sample labels
        title : str
            Plot title
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure
        """
        import plotly.figure_factory as ff
        
        fig = ff.create_dendrogram(
            linkage_matrix,
            labels=labels,
            orientation='bottom'
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Samples',
            yaxis_title='Distance',
            height=600
        )
        
        return fig
    
    def create_wordcloud(
        self,
        text: Union[str, List[str], Dict[str, float]],
        title: str = "Word Cloud",
        max_words: int = 100,
        background_color: str = 'white',
        colormap: str = 'viridis',
        width: int = 800,
        height: int = 400,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create word cloud visualization.
        
        Parameters:
        -----------
        text : str, list, or dict
            Text data or word frequencies
        title : str
            Plot title
        max_words : int
            Maximum number of words
        background_color : str
            Background color
        colormap : str
            Color map for words
        width : int
            Image width
        height : int
            Image height
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Prepare text
        if isinstance(text, list):
            text = ' '.join(text)
        elif isinstance(text, dict):
            # Use frequencies directly
            wc = WordCloud(
                width=width,
                height=height,
                background_color=background_color,
                colormap=colormap,
                max_words=max_words
            ).generate_from_frequencies(text)
        
        if not isinstance(text, dict):
            wc = WordCloud(
                width=width,
                height=height,
                background_color=background_color,
                colormap=colormap,
                max_words=max_words,
                collocations=False
            ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cluster_scatter(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "Cluster Visualization",
        xlabel: str = "Component 1",
        ylabel: str = "Component 2",
        centers: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scatter plot for clusters (2D).
        
        Parameters:
        -----------
        X : np.ndarray
            2D feature matrix
        labels : np.ndarray
            Cluster labels
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        centers : np.ndarray, optional
            Cluster centers
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot points
        scatter = ax.scatter(
            X[:, 0], X[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Plot centers if provided
        if centers is not None:
            ax.scatter(
                centers[:, 0], centers[:, 1],
                c='red',
                marker='X',
                s=200,
                edgecolors='k',
                linewidth=2,
                label='Centers'
            )
            ax.legend()
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sentiment_distribution(
        self,
        sentiments: pd.DataFrame,
        title: str = "Sentiment Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization for sentiment distribution.
        
        Parameters:
        -----------
        sentiments : pd.DataFrame
            Sentiment scores DataFrame
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sentiment label distribution
        if 'sentiment' in sentiments.columns:
            sentiments['sentiment'].value_counts().plot(
                kind='bar',
                ax=axes[0],
                color=['green', 'gray', 'red']
            )
            axes[0].set_title('Sentiment Labels')
            axes[0].set_xlabel('Sentiment')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=0)
        
        # Compound score distribution
        if 'compound' in sentiments.columns:
            axes[1].hist(sentiments['compound'], bins=30, color='steelblue', edgecolor='black')
            axes[1].axvline(0, color='red', linestyle='--', label='Neutral')
            axes[1].set_title('Compound Score Distribution')
            axes[1].set_xlabel('Compound Score')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_topic_distribution(
        self,
        topic_summary: pd.DataFrame,
        title: str = "Topic Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization for topic distribution.
        
        Parameters:
        -----------
        topic_summary : pd.DataFrame
            Topic summary DataFrame
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'n_documents' in topic_summary.columns:
            ax.bar(
                topic_summary['topic_id'],
                topic_summary['n_documents'],
                color='steelblue',
                edgecolor='black'
            )
            ax.set_xlabel('Topic ID')
            ax.set_ylabel('Number of Documents')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        title: str = "Heatmap",
        xlabel: str = "",
        ylabel: str = "",
        cmap: str = 'coolwarm',
        annot: bool = True,
        fmt: str = '.2f',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap visualization.
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            Data to visualize
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        cmap : str
            Color map
        annot : bool
            Annotate cells with values
        fmt : str
            Format string for annotations
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
            cbar_kws={'label': 'Value'}
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Made with Bob
