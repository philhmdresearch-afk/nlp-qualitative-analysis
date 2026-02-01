"""
Advanced Visualizations for UX Research
Specialized plots for theme analysis, contrastive analysis, and insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class AdvancedVisualizer:
    """
    Advanced visualizations for UX research analysis.
    
    Features:
    - Theme stability plots
    - Theme overlap heatmaps
    - Salience vs emotional weight scatterplots
    - Contrastive analysis charts
    - Participation imbalance visualizations
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize advanced visualizer.
        
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
    
    def plot_theme_stability(
        self,
        stability_results: Dict,
        topic_terms: Dict[int, List[str]]
    ) -> matplotlib.figure.Figure:
        """
        Plot theme stability scores with robustness indicators.
        
        Parameters:
        -----------
        stability_results : Dict
            Results from ThemeAnalyzer.compute_theme_stability()
        topic_terms : Dict
            Top terms for each topic
            
        Returns:
        --------
        matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        topics = []
        stability_scores = []
        robustness_labels = []
        colors = []
        
        color_map = {
            'Very Robust': '#2ecc71',
            'Robust': '#3498db',
            'Moderately Stable': '#f39c12',
            'Unstable': '#e74c3c'
        }
        
        for topic_id, metrics in stability_results.items():
            terms = ", ".join(topic_terms.get(topic_id, [])[:3])
            topics.append(f"T{topic_id}: {terms[:30]}")
            stability_scores.append(metrics['stability_score'])
            robustness_labels.append(metrics['robustness_label'])
            colors.append(color_map.get(metrics['robustness_label'], '#95a5a6'))
        
        # Plot 1: Stability scores bar chart
        bars = ax1.barh(topics, stability_scores, color=colors)
        ax1.set_xlabel('Stability Score', fontsize=12, fontweight='bold')
        ax1.set_title('Theme Stability Analysis', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Add threshold lines
        ax1.axvline(x=0.85, color='green', linestyle='--', alpha=0.5, label='Very Robust')
        ax1.axvline(x=0.70, color='blue', linestyle='--', alpha=0.5, label='Robust')
        ax1.axvline(x=0.50, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax1.legend(loc='lower right', fontsize=9)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, stability_scores)):
            ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', va='center', fontsize=9)
        
        # Plot 2: Robustness distribution
        robustness_counts = pd.Series(robustness_labels).value_counts()
        colors_pie = [color_map.get(str(label), '#95a5a6') for label in robustness_counts.index]
        
        ax2.pie(robustness_counts.values, labels=robustness_counts.index, 
               autopct='%1.0f%%', colors=colors_pie, startangle=90)
        ax2.set_title('Theme Robustness Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_theme_overlap_matrix(
        self,
        overlap_matrix: np.ndarray,
        topic_terms: Dict[int, List[str]],
        fuzziness_scores: Optional[Dict] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot theme overlap heatmap with fuzziness indicators.
        
        Parameters:
        -----------
        overlap_matrix : np.ndarray
            Theme-theme overlap matrix
        topic_terms : Dict
            Top terms for each topic
        fuzziness_scores : Dict, optional
            Fuzziness scores per theme
            
        Returns:
        --------
        matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create labels
        n_topics = overlap_matrix.shape[0]
        labels = [f"T{i}: {', '.join(topic_terms.get(i, [])[:2])}" for i in range(n_topics)]
        
        # Plot heatmap
        sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Overlap Score'}, vmin=0, vmax=1)
        
        ax.set_title('Theme Overlap Matrix\n(Higher values = more boundary documents)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add fuzziness annotations if available
        if fuzziness_scores:
            for i, (topic_id, metrics) in enumerate(fuzziness_scores.items()):
                clarity = metrics['clarity']
                color = 'green' if clarity == 'Clear' else 'orange' if clarity == 'Moderate' else 'red'
                ax.text(n_topics + 0.5, i + 0.5, f"●", color=color, fontsize=20, 
                       ha='center', va='center')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Clear'),
                Patch(facecolor='orange', label='Moderate'),
                Patch(facecolor='red', label='Fuzzy')
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(1.15, 1), title='Theme Clarity')
        
        plt.tight_layout()
        return fig
    
    def plot_salience_emotional_weight(
        self,
        plot_data: pd.DataFrame
    ) -> matplotlib.figure.Figure:
        """
        Create salience vs emotional weight scatterplot with quadrants.
        
        Parameters:
        -----------
        plot_data : pd.DataFrame
            Data from ContrastiveAnalyzer.create_salience_emotional_plot_data()
            
        Returns:
        --------
        matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define quadrant colors
        quadrant_colors = {
            'Loud & Emotional': '#e74c3c',
            'Quiet but Painful': '#f39c12',
            'Core Strengths': '#2ecc71',
            'Background Noise': '#95a5a6'
        }
        
        # Plot points
        for quadrant in plot_data['quadrant'].unique():
            data = plot_data[plot_data['quadrant'] == quadrant]
            ax.scatter(data['salience'], data['emotional_weight'], 
                      c=quadrant_colors.get(quadrant, '#95a5a6'),
                      s=200, alpha=0.6, label=quadrant, edgecolors='black', linewidth=1)
        
        # Add labels for each point
        for _, row in plot_data.iterrows():
            ax.annotate(str(row['label']),
                       (float(row['salience']), float(row['emotional_weight'])),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        # Add quadrant lines
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.15, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.05, 0.45, 'Quiet but\nPainful', fontsize=11, 
               ha='center', va='center', alpha=0.5, fontweight='bold')
        ax.text(0.25, 0.45, 'Loud &\nEmotional', fontsize=11, 
               ha='center', va='center', alpha=0.5, fontweight='bold')
        ax.text(0.05, 0.15, 'Background\nNoise', fontsize=11, 
               ha='center', va='center', alpha=0.5, fontweight='bold')
        ax.text(0.25, 0.15, 'Core\nStrengths', fontsize=11, 
               ha='center', va='center', alpha=0.5, fontweight='bold')
        
        ax.set_xlabel('Salience (Prevalence)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Emotional Weight (Sentiment Intensity)', fontsize=12, fontweight='bold')
        ax.set_title('Theme Prioritization Matrix\nSalience vs Emotional Weight', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_contrastive_themes(
        self,
        contrastive_summary: Dict,
        top_n: int = 5
    ) -> matplotlib.figure.Figure:
        """
        Plot contrastive theme comparison between groups.
        
        Parameters:
        -----------
        contrastive_summary : Dict
            Results from ContrastiveAnalyzer.generate_contrastive_summary()
        top_n : int
            Number of top themes to show per group
            
        Returns:
        --------
        matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        group_a = contrastive_summary['group_a']
        group_b = contrastive_summary['group_b']
        
        # Plot distinctive themes for group A
        distinctive_a = contrastive_summary['distinctive_to_a'][:top_n]
        if distinctive_a:
            terms_a = [d['terms'][:30] for d in distinctive_a]
            prevalence_a = [d['prevalence'] * 100 for d in distinctive_a]
            
            bars_a = ax1.barh(terms_a, prevalence_a, color='#3498db', alpha=0.7)
            ax1.set_xlabel('Prevalence (%)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Distinctive to {group_a}', fontsize=13, fontweight='bold')
            ax1.set_xlim(0, max(prevalence_a) * 1.2)
            
            # Add value labels
            for bar, val in zip(bars_a, prevalence_a):
                ax1.text(val + 1, bar.get_y() + bar.get_height()/2, 
                        f'{val:.1f}%', va='center', fontsize=9)
        
        # Plot distinctive themes for group B
        distinctive_b = contrastive_summary['distinctive_to_b'][:top_n]
        if distinctive_b:
            terms_b = [d['terms'][:30] for d in distinctive_b]
            prevalence_b = [d['prevalence'] * 100 for d in distinctive_b]
            
            bars_b = ax2.barh(terms_b, prevalence_b, color='#e74c3c', alpha=0.7)
            ax2.set_xlabel('Prevalence (%)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Distinctive to {group_b}', fontsize=13, fontweight='bold')
            ax2.set_xlim(0, max(prevalence_b) * 1.2)
            
            # Add value labels
            for bar, val in zip(bars_b, prevalence_b):
                ax2.text(val + 1, bar.get_y() + bar.get_height()/2, 
                        f'{val:.1f}%', va='center', fontsize=9)
        
        fig.suptitle(f'Contrastive Theme Analysis: {group_a} vs {group_b}', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_theme_sentiment_interaction(
        self,
        interactions: Dict,
        topic_terms: Dict[int, List[str]]
    ) -> matplotlib.figure.Figure:
        """
        Plot Theme × Sentiment interactions across groups.
        
        Parameters:
        -----------
        interactions : Dict
            Results from ContrastiveAnalyzer.analyze_theme_sentiment_interaction()
        topic_terms : Dict
            Top terms for each topic
            
        Returns:
        --------
        matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        themes = []
        groups = set()
        sentiment_data = []
        
        for topic_id, data in interactions.items():
            terms = ", ".join(topic_terms.get(topic_id, [])[:3])
            themes.append(f"T{topic_id}: {terms[:25]}")
            
            for group, metrics in data['group_sentiments'].items():
                groups.add(group)
                sentiment_data.append({
                    'theme': f"T{topic_id}: {terms[:25]}",
                    'group': group,
                    'sentiment': metrics['avg_sentiment']
                })
        
        # Create grouped bar chart
        df = pd.DataFrame(sentiment_data)
        groups_list = sorted(list(groups))
        
        x = np.arange(len(themes))
        width = 0.8 / len(groups_list)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, group in enumerate(groups_list):
            group_data = df[df['group'] == group]
            sentiments = []
            for theme in themes:
                theme_data = group_data[group_data['theme'] == theme]
                if len(theme_data) > 0:
                    # Get sentiment value safely - type: ignore for pandas operations
                    try:
                        sent_col = theme_data['sentiment']  # type: ignore
                        if hasattr(sent_col, 'to_numpy'):
                            sent_val = sent_col.to_numpy()[0]  # type: ignore
                        else:
                            sent_val = np.array(sent_col)[0]
                        sentiments.append(float(sent_val))
                    except (IndexError, AttributeError, TypeError):
                        sentiments.append(0.0)
                else:
                    sentiments.append(0.0)
            
            offset = (i - len(groups_list)/2 + 0.5) * width
            bars = ax.bar(x + offset, sentiments, width, label=group, 
                         color=colors[i % len(colors)], alpha=0.8)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Styling
        ax.set_xlabel('Themes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Sentiment', fontsize=12, fontweight='bold')
        ax.set_title('Theme × Sentiment Interaction Across Groups', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(themes, rotation=45, ha='right')
        ax.legend(title='Group', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_participation_imbalance(
        self,
        contribution_data: Dict
    ) -> matplotlib.figure.Figure:
        """
        Plot participation imbalance warnings.
        
        Parameters:
        -----------
        contribution_data : Dict
            Results from QuoteAnalyzer.track_participant_contributions()
            
        Returns:
        --------
        matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Contribution distribution
        contributions = contribution_data['contribution_distribution']
        participants = list(contributions.keys())
        counts = list(contributions.values())
        
        # Sort by count
        sorted_data = sorted(zip(participants, counts), key=lambda x: x[1], reverse=True)
        participants_sorted = [str(p) for p, _ in sorted_data[:15]]  # Top 15
        counts_sorted = [c for _, c in sorted_data[:15]]
        
        colors = ['#e74c3c' if p in contribution_data.get('dominant_participants', {}) 
                 else '#3498db' for p in participants_sorted]
        
        bars = ax1.bar(range(len(participants_sorted)), counts_sorted, color=colors, alpha=0.7)
        ax1.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Contributions', fontsize=12, fontweight='bold')
        ax1.set_title('Participation Distribution\n(Red = Dominant Participants)', 
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(participants_sorted)))
        ax1.set_xticklabels(participants_sorted, rotation=45, ha='right')
        
        # Add average line
        avg_contributions = contribution_data['avg_contributions_per_participant']
        ax1.axhline(y=avg_contributions, color='green', linestyle='--', 
                   label=f'Average: {avg_contributions:.1f}', linewidth=2)
        ax1.legend()
        
        # Plot 2: Gini coefficient visualization
        gini = contribution_data.get('gini_coefficient', 0)
        
        # Create Lorenz curve
        sorted_contributions = sorted(counts)
        cumsum = np.cumsum(sorted_contributions)
        cumsum_pct = cumsum / cumsum[-1] * 100
        participants_pct = np.arange(1, len(sorted_contributions) + 1) / len(sorted_contributions) * 100
        
        ax2.plot(participants_pct, cumsum_pct, 'b-', linewidth=2, label='Lorenz Curve')
        ax2.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Equality')
        ax2.fill_between(participants_pct, cumsum_pct, participants_pct, alpha=0.3)
        
        ax2.set_xlabel('Cumulative % of Participants', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative % of Contributions', fontsize=12, fontweight='bold')
        ax2.set_title(f'Participation Inequality\nGini Coefficient: {gini:.3f}', 
                     fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_emergent_vs_dominant(
        self,
        theme_categories: Dict,
        topic_terms: Dict[int, List[str]]
    ) -> matplotlib.figure.Figure:
        """
        Plot emergent vs dominant themes.
        
        Parameters:
        -----------
        theme_categories : Dict
            Results from ThemeAnalyzer.identify_emergent_themes()
        topic_terms : Dict
            Top terms for each topic
            
        Returns:
        --------
        matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        categories = []
        frequencies = []
        intensities = []
        labels = []
        colors_map = {
            'Dominant Theme': '#2ecc71',
            'Emergent Theme (Weak Signal)': '#f39c12',
            'Background Theme': '#3498db',
            'Noise': '#95a5a6'
        }
        colors = []
        
        for topic_id, data in theme_categories.items():
            terms = ", ".join(topic_terms.get(topic_id, [])[:2])
            labels.append(f"T{topic_id}: {terms[:20]}")
            categories.append(data['category'])
            frequencies.append(data['frequency'])
            intensities.append(data['intensity'])
            colors.append(colors_map.get(data['category'], '#95a5a6'))
        
        # Create scatter plot
        scatter = ax.scatter(frequencies, intensities, c=colors, s=300, alpha=0.6, 
                           edgecolors='black', linewidth=1.5)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (frequencies[i], intensities[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        # Add quadrant lines
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Intensity threshold')
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Frequency threshold')
        
        # Add quadrant labels
        ax.text(0.05, 0.85, 'Emergent\n(Weak Signal)', fontsize=11, 
               ha='center', va='center', alpha=0.5, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))
        ax.text(0.25, 0.85, 'Dominant\nTheme', fontsize=11, 
               ha='center', va='center', alpha=0.5, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
        
        ax.set_xlabel('Frequency (Prevalence)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intensity (Strength when present)', fontsize=12, fontweight='bold')
        ax.set_title('Emergent vs Dominant Themes', fontsize=14, fontweight='bold')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat) 
                          for cat, color in colors_map.items()]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Made with Bob
