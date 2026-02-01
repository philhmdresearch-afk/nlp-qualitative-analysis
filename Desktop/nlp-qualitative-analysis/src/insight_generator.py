"""
Insight Generation Module for UX Research
Generate narrative-ready insights with evidence and stakeholder views
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class InsightGenerator:
    """
    Generate narrative-ready insights for stakeholders.
    
    Features:
    - Auto-generate short, memorable insight statements (~50 words)
    - Context + Problem → Actionable Insight format
    - Evidence bundling
    - Stakeholder-specific views
    - Longitudinal pattern detection
    - Analyst annotation layer
    """
    
    def __init__(self):
        """Initialize insight generator."""
        self.insights = []
        self.annotations = {}
        
    def generate_insight_statement(
        self,
        topic_id: int,
        topic_terms: List[str],
        prevalence: float,
        sentiment: Optional[float] = None,
        group_differences: Optional[Dict] = None,
        stability_score: Optional[float] = None,
        supporting_quotes: Optional[List[Dict]] = None,
        n_participants: Optional[int] = None
    ) -> Dict:
        """
        Generate a narrative insight statement with supporting evidence.
        
        Parameters:
        -----------
        topic_id : int
            Topic identifier
        topic_terms : List[str]
            Top terms for this topic
        prevalence : float
            Topic prevalence (0-1)
        sentiment : float, optional
            Average sentiment for this topic
        group_differences : Dict, optional
            Differences across groups
        stability_score : float, optional
            Theme stability score
        supporting_quotes : List[Dict], optional
            Supporting quotes with metadata
        n_participants : int, optional
            Number of participants mentioning this theme
            
        Returns:
        --------
        Dict with insight statement and evidence bundle
        """
        # Generate headline (Context + Problem)
        headline = self._generate_headline(
            topic_terms, prevalence, sentiment, group_differences
        )
        
        # Generate description (Context + Problem → Actionable Insight, ~50 words)
        description = self._generate_description(
            topic_terms, prevalence, sentiment, group_differences, n_participants
        )
        
        # Determine confidence level
        confidence = self._assess_confidence(
            prevalence, stability_score, n_participants
        )
        
        # Determine priority
        priority = self._assess_priority(
            prevalence, sentiment, group_differences
        )
        
        # Create evidence bundle
        evidence = {
            'theme_terms': topic_terms,
            'prevalence': prevalence,
            'prevalence_pct': prevalence * 100,
            'n_participants': n_participants,
            'sentiment': sentiment,
            'stability_score': stability_score,
            'supporting_quotes': supporting_quotes or [],
            'group_differences': group_differences
        }
        
        # Generate implications
        implications = self._generate_implications(
            topic_terms, prevalence, sentiment, group_differences
        )
        
        insight = {
            'topic_id': topic_id,
            'headline': headline,
            'description': description,
            'confidence': confidence,
            'priority': priority,
            'evidence': evidence,
            'implications': implications,
            'generated_at': datetime.now().isoformat(),
            'editable': True,
            'analyst_notes': None
        }
        
        self.insights.append(insight)
        return insight
    
    def _generate_headline(
        self,
        terms: List[str],
        prevalence: float,
        sentiment: Optional[float],
        group_differences: Optional[Dict]
    ) -> str:
        """Generate short, memorable insight headline (Context+Problem)."""
        main_theme = terms[0] if terms else "theme"
        
        # Context + Problem format
        if group_differences and len(group_differences) > 0:
            return f"User groups disagree on {main_theme}"
        elif sentiment is not None and sentiment < -0.1:
            return f"Users frustrated with {main_theme}"
        elif sentiment is not None and sentiment > 0.1:
            return f"Users value {main_theme}"
        elif prevalence > 0.3:
            return f"{main_theme.capitalize()} is a top priority"
        else:
            return f"{main_theme.capitalize()} emerging as concern"
    
    def _generate_description(
        self,
        terms: List[str],
        prevalence: float,
        sentiment: Optional[float],
        group_differences: Optional[Dict],
        n_participants: Optional[int]
    ) -> str:
        """
        Generate short, memorable insight (~50 words).
        Format: Context + Problem → Actionable Insight
        """
        main_theme = terms[0] if terms else "this theme"
        secondary_terms = ", ".join(terms[1:3]) if len(terms) > 1 else ""
        
        # Context + Problem
        context = f"{prevalence*100:.0f}% of users"
        if n_participants:
            context += f" ({n_participants} participants)"
        
        if sentiment is not None and sentiment < -0.1:
            problem = f"express frustration with {main_theme}"
            if secondary_terms:
                problem += f", particularly around {secondary_terms}"
        elif sentiment is not None and sentiment > 0.1:
            problem = f"appreciate {main_theme}"
            if secondary_terms:
                problem += f", especially {secondary_terms}"
        else:
            problem = f"discuss {main_theme}"
            if secondary_terms:
                problem += f" and {secondary_terms}"
        
        # Actionable insight
        if sentiment is not None and sentiment < -0.1:
            action = f"Prioritize improvements to {main_theme} to address user pain points"
        elif group_differences:
            action = f"Tailor {main_theme} experience for different user segments"
        elif prevalence > 0.25:
            action = f"Invest in {main_theme} as it's a key user priority"
        else:
            action = f"Monitor {main_theme} as an emerging user need"
        
        # Combine: Context + Problem → Action (target ~50 words)
        insight = f"{context} {problem}. {action}."
        
        return insight
    
    def _assess_confidence(
        self,
        prevalence: float,
        stability_score: Optional[float],
        n_participants: Optional[int]
    ) -> str:
        """Assess confidence level in the insight."""
        confidence_score = 0
        
        # Prevalence contributes to confidence
        if prevalence > 0.2:
            confidence_score += 2
        elif prevalence > 0.1:
            confidence_score += 1
        
        # Stability contributes to confidence
        if stability_score is not None:
            if stability_score > 0.7:
                confidence_score += 2
            elif stability_score > 0.5:
                confidence_score += 1
        
        # Number of participants contributes
        if n_participants is not None:
            if n_participants > 10:
                confidence_score += 2
            elif n_participants > 5:
                confidence_score += 1
        
        if confidence_score >= 5:
            return "High"
        elif confidence_score >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _assess_priority(
        self,
        prevalence: float,
        sentiment: Optional[float],
        group_differences: Optional[Dict]
    ) -> str:
        """Assess priority level."""
        priority_score = 0
        
        # High prevalence = higher priority
        if prevalence > 0.25:
            priority_score += 2
        elif prevalence > 0.15:
            priority_score += 1
        
        # Negative sentiment = higher priority
        if sentiment is not None and sentiment < -0.1:
            priority_score += 2
        
        # Group differences = higher priority
        if group_differences and len(group_differences) > 0:
            priority_score += 1
        
        if priority_score >= 4:
            return "Critical"
        elif priority_score >= 2:
            return "High"
        else:
            return "Medium"
    
    def _generate_implications(
        self,
        terms: List[str],
        prevalence: float,
        sentiment: Optional[float],
        group_differences: Optional[Dict]
    ) -> List[str]:
        """Generate actionable implications."""
        implications = []
        main_theme = ", ".join(terms[:2])
        
        if sentiment is not None and sentiment < -0.1:
            implications.append(f"Address user concerns regarding {main_theme}")
            implications.append("Consider prioritizing improvements in this area")
        
        if prevalence > 0.25:
            implications.append(f"High visibility of {main_theme} suggests it's a key user priority")
        
        if group_differences:
            implications.append("Tailor solutions to different user segment needs")
        
        return implications
    
    def create_stakeholder_view(
        self,
        insights: List[Dict],
        stakeholder_type: str
    ) -> Dict:
        """
        Create stakeholder-specific view of insights.
        
        Parameters:
        -----------
        insights : List[Dict]
            List of generated insights
        stakeholder_type : str
            'product', 'design', 'leadership', or 'research'
            
        Returns:
        --------
        Dict with filtered and formatted insights
        """
        if stakeholder_type == 'product':
            return self._product_view(insights)
        elif stakeholder_type == 'design':
            return self._design_view(insights)
        elif stakeholder_type == 'leadership':
            return self._leadership_view(insights)
        else:
            return self._research_view(insights)
    
    def _product_view(self, insights: List[Dict]) -> Dict:
        """Product manager view: feature implications."""
        # Focus on high-priority insights with clear implications
        relevant_insights = [
            i for i in insights
            if i['priority'] in ['Critical', 'High']
        ]
        
        formatted = []
        for insight in relevant_insights:
            formatted.append({
                'headline': insight['headline'],
                'insight': insight['description'],
                'impact': insight['priority'],
                'affected_users': f"{insight['evidence']['prevalence_pct']:.0f}%",
                'feature_implications': insight['implications'],
                'sentiment': insight['evidence'].get('sentiment'),
                'action_required': insight['priority'] == 'Critical'
            })
        
        return {
            'stakeholder': 'Product Management',
            'focus': 'Feature-level implications and user impact',
            'insights': formatted,
            'summary': f"{len(formatted)} high-priority items requiring attention"
        }
    
    def _design_view(self, insights: List[Dict]) -> Dict:
        """Design view: pain points and unmet needs."""
        # Focus on negative sentiment and user friction
        relevant_insights = [
            i for i in insights
            if i['evidence'].get('sentiment', 0) < 0 or 'frustrat' in i['headline'].lower()
        ]
        
        formatted = []
        for insight in relevant_insights:
            formatted.append({
                'pain_point': insight['headline'],
                'insight': insight['description'],
                'severity': insight['priority'],
                'user_quotes': [q['text'][:150] + "..." for q in insight['evidence']['supporting_quotes'][:2]],
                'design_opportunity': self._extract_design_opportunity(insight),
                'affected_users': f"{insight['evidence']['prevalence_pct']:.0f}%"
            })
        
        return {
            'stakeholder': 'Design Team',
            'focus': 'Pain points and design opportunities',
            'insights': formatted,
            'summary': f"{len(formatted)} pain points identified"
        }
    
    def _leadership_view(self, insights: List[Dict]) -> Dict:
        """Leadership view: risks, opportunities, trends."""
        # Focus on high-level patterns and strategic implications
        critical_insights = [i for i in insights if i['priority'] == 'Critical']
        trend_insights = [i for i in insights if i['evidence']['prevalence'] > 0.2]
        
        formatted = {
            'critical_risks': [],
            'opportunities': [],
            'key_trends': []
        }
        
        for insight in critical_insights:
            if insight['evidence'].get('sentiment', 0) < 0:
                formatted['critical_risks'].append({
                    'issue': insight['headline'],
                    'insight': insight['description'],
                    'impact': f"{insight['evidence']['prevalence_pct']:.0f}% of users",
                    'confidence': insight['confidence']
                })
            else:
                formatted['opportunities'].append({
                    'opportunity': insight['headline'],
                    'insight': insight['description'],
                    'potential': f"{insight['evidence']['prevalence_pct']:.0f}% user interest"
                })
        
        for insight in trend_insights:
            formatted['key_trends'].append({
                'trend': insight['headline'],
                'insight': insight['description'],
                'prevalence': f"{insight['evidence']['prevalence_pct']:.0f}%",
                'direction': 'Positive' if insight['evidence'].get('sentiment', 0) > 0 else 'Negative'
            })
        
        return {
            'stakeholder': 'Leadership',
            'focus': 'Strategic risks, opportunities, and trends',
            'insights': formatted,
            'executive_summary': self._generate_executive_summary(insights)
        }
    
    def _research_view(self, insights: List[Dict]) -> Dict:
        """Research view: full methodological detail."""
        return {
            'stakeholder': 'Research Team',
            'focus': 'Complete analysis with methodological details',
            'insights': insights,
            'methodology_notes': {
                'confidence_assessment': 'Based on prevalence, stability, and participant count',
                'priority_assessment': 'Based on prevalence, sentiment, and group differences',
                'limitations': 'Insights are model-derived and should be validated with domain expertise'
            }
        }
    
    def _extract_design_opportunity(self, insight: Dict) -> str:
        """Extract design opportunity from insight."""
        terms = insight['evidence']['theme_terms'][:2]
        if insight['evidence'].get('sentiment', 0) < -0.1:
            return f"Redesign {terms[0]} experience to address user frustrations"
        else:
            return f"Enhance {terms[0]} to better meet user needs"
    
    def _generate_executive_summary(self, insights: List[Dict]) -> str:
        """Generate executive summary."""
        n_critical = sum(1 for i in insights if i['priority'] == 'Critical')
        n_high = sum(1 for i in insights if i['priority'] == 'High')
        
        avg_sentiment = np.mean([
            i['evidence'].get('sentiment', 0)
            for i in insights
            if i['evidence'].get('sentiment') is not None
        ])
        
        summary = f"Analysis identified {n_critical} critical issues and {n_high} high-priority items. "
        
        if avg_sentiment < -0.05:
            summary += "Overall user sentiment is negative, indicating areas requiring immediate attention."
        elif avg_sentiment > 0.05:
            summary += "Overall user sentiment is positive, with opportunities to build on strengths."
        else:
            summary += "User sentiment is mixed, with both challenges and opportunities identified."
        
        return summary
    
    def add_analyst_annotation(
        self,
        insight_id: int,
        annotation_type: str,
        content: str,
        analyst_name: Optional[str] = None
    ) -> Dict:
        """
        Add analyst annotation to an insight.
        
        Parameters:
        -----------
        insight_id : int
            Index of insight to annotate
        annotation_type : str
            'rename', 'merge', 'split', 'memo', 'override'
        content : str
            Annotation content
        analyst_name : str, optional
            Name of analyst making annotation
            
        Returns:
        --------
        Dict with annotation details
        """
        if insight_id >= len(self.insights):
            return {'error': 'Invalid insight_id'}
        
        annotation = {
            'type': annotation_type,
            'content': content,
            'analyst': analyst_name,
            'timestamp': datetime.now().isoformat(),
            'original_insight': self.insights[insight_id].copy()
        }
        
        # Store annotation
        if insight_id not in self.annotations:
            self.annotations[insight_id] = []
        self.annotations[insight_id].append(annotation)
        
        # Apply annotation
        if annotation_type == 'rename':
            self.insights[insight_id]['headline'] = content
            self.insights[insight_id]['analyst_modified'] = True
        elif annotation_type == 'memo':
            self.insights[insight_id]['analyst_notes'] = content
        
        return annotation
    
    def detect_longitudinal_patterns(
        self,
        historical_insights: List[Dict],
        current_insights: List[Dict],
        time_periods: List[str]
    ) -> Dict:
        """
        Detect patterns across multiple time periods.
        
        Parameters:
        -----------
        historical_insights : List[Dict]
            Insights from previous time periods
        current_insights : List[Dict]
            Current insights
        time_periods : List[str]
            Time period labels
            
        Returns:
        --------
        Dict with longitudinal patterns
        """
        patterns = {
            'emerging_themes': [],
            'declining_themes': [],
            'persistent_themes': [],
            'sentiment_shifts': []
        }
        
        # Compare current vs historical
        historical_themes = {
            i['topic_id']: i['evidence']['prevalence']
            for i in historical_insights
        }
        
        current_themes = {
            i['topic_id']: i['evidence']['prevalence']
            for i in current_insights
        }
        
        # Identify emerging themes (new or growing)
        for topic_id, prevalence in current_themes.items():
            if topic_id not in historical_themes:
                insight = next(i for i in current_insights if i['topic_id'] == topic_id)
                patterns['emerging_themes'].append({
                    'theme': insight['headline'],
                    'current_prevalence': prevalence,
                    'status': 'New theme'
                })
            elif prevalence > historical_themes[topic_id] * 1.5:
                insight = next(i for i in current_insights if i['topic_id'] == topic_id)
                patterns['emerging_themes'].append({
                    'theme': insight['headline'],
                    'growth': (prevalence - historical_themes[topic_id]) / historical_themes[topic_id] * 100,
                    'status': 'Growing'
                })
        
        # Identify declining themes
        for topic_id, prev_prevalence in historical_themes.items():
            if topic_id in current_themes:
                curr_prevalence = current_themes[topic_id]
                if curr_prevalence < prev_prevalence * 0.5:
                    insight = next((i for i in current_insights if i['topic_id'] == topic_id), None)
                    if insight:
                        patterns['declining_themes'].append({
                            'theme': insight['headline'],
                            'decline': (prev_prevalence - curr_prevalence) / prev_prevalence * 100,
                            'status': 'Declining'
                        })
        
        return patterns
    
    def export_insights(self, format: str = 'markdown') -> str:
        """
        Export insights in various formats.
        
        Parameters:
        -----------
        format : str
            'markdown', 'html', or 'json'
            
        Returns:
        --------
        Formatted string
        """
        if format == 'markdown':
            return self._export_markdown()
        elif format == 'html':
            return self._export_html()
        else:
            import json
            return json.dumps(self.insights, indent=2)
    
    def _export_markdown(self) -> str:
        """Export as markdown."""
        md = "# Research Insights\n\n"
        md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        for i, insight in enumerate(self.insights, 1):
            md += f"## {i}. {insight['headline']}\n\n"
            md += f"**Priority:** {insight['priority']} | **Confidence:** {insight['confidence']}\n\n"
            md += f"**Insight:** {insight['description']}\n\n"
            
            if insight['implications']:
                md += "**Implications:**\n"
                for impl in insight['implications']:
                    md += f"- {impl}\n"
                md += "\n"
            
            if insight['evidence']['supporting_quotes']:
                md += "**Supporting Evidence:**\n"
                for quote in insight['evidence']['supporting_quotes'][:2]:
                    md += f"> {quote['text'][:150]}...\n\n"
            
            md += "---\n\n"
        
        return md
    
    def _export_html(self) -> str:
        """Export as HTML."""
        html = "<html><head><title>Research Insights</title></head><body>"
        html += "<h1>Research Insights</h1>"
        
        for insight in self.insights:
            html += f"<div class='insight'>"
            html += f"<h2>{insight['headline']}</h2>"
            html += f"<p><strong>Priority:</strong> {insight['priority']} | "
            html += f"<strong>Confidence:</strong> {insight['confidence']}</p>"
            html += f"<p><strong>Insight:</strong> {insight['description']}</p>"
            html += "</div>"
        
        html += "</body></html>"
        return html

# Made with Bob
