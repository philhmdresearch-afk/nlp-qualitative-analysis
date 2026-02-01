# Advanced UX Research Features

## Overview

This document describes the advanced qualitative analysis features designed specifically for UX researchers. These features go beyond basic topic modeling to provide **sense-making outputs**, **traceability**, and **narrative-ready insights** that address real-world UX research needs.

---

## 🎯 Core Philosophy

These features are designed to:
1. **Help UXRs think, not just report** - Analytical affordances over raw statistics
2. **Build trust through rigor** - Methodological transparency and robustness indicators
3. **Bridge analysis to storytelling** - Natural language insights ready for stakeholders
4. **Preserve qualitative epistemology** - Analyst control and annotation layers
5. **Prevent deskilling** - Tools augment, not replace, researcher judgment

---

## 📊 Feature Categories

### 1. Sense-Making Outputs

#### 1.1 Theme Stability & Robustness Indicators

**Module:** `src/theme_analysis.py` → `ThemeAnalyzer.compute_theme_stability()`

**What it does:**
- Runs bootstrap resampling to test if themes persist across data variations
- Computes stability scores (0-1, higher = more robust)
- Labels themes as "Very Robust", "Robust", "Moderately Stable", or "Unstable"

**Why UXRs need it:**
- Addresses the anxiety: "Are these themes real, or just model noise?"
- Provides permission to say "this finding is robust" in readouts
- Signals methodological rigor without statistical intimidation

**Example output:**
```python
{
    0: {
        'stability_score': 0.87,
        'robustness_label': 'Very Robust',
        'persistence_rate': 87.0,
        'confidence': 'High'
    }
}
```

**Visualization:** Bar chart with color-coded robustness levels

---

#### 1.2 Theme Overlap & Boundary Detection

**Module:** `src/theme_analysis.py` → `ThemeAnalyzer.compute_theme_overlap()`

**What it does:**
- Computes theme-theme overlap matrices (Jaccard similarity)
- Identifies "boundary documents" that sit between themes
- Calculates fuzziness indicators (how cleanly documents are assigned)

**Why UXRs need it:**
- Reflects how qualitative analysis actually works ("this quote fits multiple themes")
- Helps defend against false precision critique of topic models
- Shows where themes blur vs. where they're distinct

**Example output:**
```python
{
    'overlap_matrix': [[1.0, 0.3], [0.3, 1.0]],
    'boundary_documents': [5, 12, 23],
    'theme_fuzziness': {
        0: {
            'fuzziness_score': 0.25,
            'clarity': 'Clear',
            'n_boundary_docs': 3
        }
    }
}
```

**Visualization:** Heatmap with fuzziness indicators

---

#### 1.3 Emergent vs. Dominant Themes

**Module:** `src/theme_analysis.py` → `ThemeAnalyzer.identify_emergent_themes()`

**What it does:**
- Separates high-frequency themes from high-impact but low-frequency themes
- Identifies "weak signals" - rare but emotionally intense topics
- Categorizes themes as: Dominant, Emergent (Weak Signal), Background, or Noise

**Why UXRs need it:**
- Enables "early warning" narratives for product teams
- Lets teams justify acting on "small but important" signals
- Prevents overlooking critical but infrequent issues

**Example output:**
```python
{
    0: {
        'category': 'Emergent Theme (Weak Signal)',
        'priority': 'High',
        'frequency': 0.08,  # Only 8% of users
        'intensity': 0.85,  # But very strong when present
        'emotional_weight': 0.72
    }
}
```

**Visualization:** Scatterplot (frequency vs. intensity) with quadrants

---

### 2. Comparison & Difference Outputs

#### 2.1 Contrastive Theme Summaries

**Module:** `src/contrastive_analysis.py` → `ContrastiveAnalyzer.generate_contrastive_summary()`

**What it does:**
- Compares themes across groups (e.g., SMEs vs. Enterprises)
- Identifies what's distinctive to each group vs. shared
- Generates natural language contrast statements

**Why UXRs need it:**
- Bridges statistics to storytelling
- Makes analysis immediately usable in presentations
- Goes beyond "Group A differs from Group B" to explain HOW

**Example output:**
```python
{
    'natural_language_summaries': [
        "SMEs discuss automation significantly more than Enterprises (45% vs 12%)",
        "Both groups discuss onboarding, but SMEs focus on speed while Enterprises emphasize clarity"
    ],
    'distinctive_to_a': [
        {'terms': 'automation, workflow', 'prevalence': 0.45, 'difference': 0.33}
    ]
}
```

**Visualization:** Side-by-side bar charts

---

#### 2.2 Theme × Sentiment Interactions

**Module:** `src/contrastive_analysis.py` → `ContrastiveAnalyzer.analyze_theme_sentiment_interaction()`

**What it does:**
- Analyzes sentiment WITHIN specific themes, not just globally
- Detects polarity shifts (same theme, opposite sentiment across groups)
- Weights sentiment by topic probability

**Why UXRs need it:**
- Far more actionable than global sentiment
- Reveals nuanced group differences
- Example: "Automation" is +0.42 for SMEs but -0.12 for Enterprises

**Example output:**
```python
{
    0: {
        'topic_terms': 'automation, workflow',
        'group_sentiments': {
            'SMEs': {'avg_sentiment': 0.42, 'sentiment_label': 'Positive'},
            'Enterprises': {'avg_sentiment': -0.12, 'sentiment_label': 'Negative'}
        },
        'polarity_shift': 'Polarity Shift Detected'
    }
}
```

**Visualization:** Grouped bar chart showing sentiment by theme and group

---

#### 2.3 Salience vs. Emotional Weight

**Module:** `src/contrastive_analysis.py` → `ContrastiveAnalyzer.create_salience_emotional_plot_data()`

**What it does:**
- Plots themes on prevalence (X) vs. emotional intensity (Y)
- Creates four quadrants: Loud & Emotional, Quiet but Painful, Core Strengths, Noise
- Enables prioritization discussions

**Why UXRs need it:**
- Executive-friendly visualization
- Helps teams decide what to act on first
- Surfaces "quiet but painful" issues that might be missed

**Example output:**
```python
{
    'topic_id': 0,
    'salience': 0.12,  # Low frequency
    'emotional_weight': 0.68,  # High intensity
    'quadrant': 'Quiet but Painful',
    'priority': 'High'
}
```

**Visualization:** Scatterplot with quadrant labels

---

### 3. Traceability & Auditability

#### 3.1 Quote Provenance Chains

**Module:** `src/quote_analysis.py` → `QuoteAnalyzer.create_provenance_chain()`

**What it does:**
- Links each theme to specific quotes with full metadata
- Tracks: participant IDs, sessions, timestamps, tasks, context
- Shows how many contributors and sessions support each theme

**Why UXRs need it:**
- Allows defending insights under stakeholder pushback
- Reduces fear of "the AI made this up"
- Provides audit trail for research quality

**Example output:**
```python
{
    'topic_id': 0,
    'quotes': [
        {
            'text': 'The automation feature saves me hours...',
            'topic_strength': 0.87,
            'participant_id': 'P042',
            'session': 'Session_3',
            'timestamp': '2026-01-15T14:30:00'
        }
    ],
    'n_contributors': 12,
    'coverage': '12 participants across 5 sessions'
}
```

---

#### 3.2 Quote Representativeness Indicators

**Module:** `src/quote_analysis.py` → `QuoteAnalyzer.assess_quote_representativeness()`

**What it does:**
- Assesses how typical a quote is of its theme
- Labels quotes as: "Highly Representative", "Representative", "Illustrative but Uncommon", or "Edge Case"
- Computes semantic similarity to other theme documents

**Why UXRs need it:**
- Directly addresses: "Is this just one loud participant?"
- Helps select the BEST quotes for presentations
- Builds confidence in quote selection

**Example output:**
```python
{
    'representativeness': 'Highly Representative',
    'topic_strength': 0.82,
    'percentile': 85.0,  # Top 15% of theme documents
    'confidence': 'High',
    'reason': 'Strong topic association and high semantic similarity'
}
```

---

#### 3.3 Participation Imbalance Warnings

**Module:** `src/quote_analysis.py` → `QuoteAnalyzer.track_participant_contributions()`

**What it does:**
- Tracks contribution counts per participant
- Identifies dominant participants (>2x average)
- Detects if specific participants drive sentiment or topics
- Computes Gini coefficient for inequality

**Why UXRs need it:**
- Guards against over-weighting dominant voices
- Surfaces bias in data collection
- Helps improve future study designs

**Example output:**
```python
{
    'gini_coefficient': 0.42,  # 0 = perfect equality, 1 = perfect inequality
    'dominant_participants': {'P007': 45},  # 45 contributions
    'sentiment_warnings': [
        {
            'participant_id': 'P007',
            'warning': 'Participant P007 contributes 65% negative sentiment'
        }
    ],
    'topic_concentration_warnings': [
        {
            'topic_id': 2,
            'warning': 'Topic 2: 55% from participant P007'
        }
    ]
}
```

**Visualization:** Bar chart + Lorenz curve with Gini coefficient

---

#### 3.4 Question-Driven Bias Detection

**Module:** `src/quote_analysis.py` → `QuoteAnalyzer.detect_question_bias()`

**What it does:**
- Detects if certain questions drive specific themes or sentiment
- Identifies leading questions
- Provides recommendations for question refinement

**Why UXRs need it:**
- Gold for senior UXRs refining study protocols
- Helps improve interview guides
- Surfaces methodological issues

**Example output:**
```python
{
    'question_analysis': {
        'What frustrates you most?': {
            'avg_sentiment': -0.45,
            'bias_warning': 'This question may elicit negative responses',
            'dominant_topics': [2, 5]
        }
    },
    'recommendations': [
        "Consider rephrasing: 'What frustrates you most?' - may elicit negative responses"
    ]
}
```

---

### 4. Narrative-Ready Outputs

#### 4.1 Insight Statement Generator

**Module:** `src/insight_generator.py` → `InsightGenerator.generate_insight_statement()`

**What it does:**
- Auto-generates short (~50 words), memorable insights
- Format: **Context + Problem → Actionable Insight**
- Includes confidence and priority levels
- Bundles supporting evidence

**Why UXRs need it:**
- Prevents endless reworking of outputs
- Provides starting point for reports
- Ensures consistent insight format

**Example output:**
```python
{
    'headline': 'Users frustrated with automation',
    'description': '35% of users (12 participants) express frustration with automation, particularly around workflow integration. Prioritize improvements to automation to address user pain points.',
    'confidence': 'High',
    'priority': 'Critical',
    'evidence': {
        'prevalence_pct': 35.0,
        'sentiment': -0.32,
        'supporting_quotes': [...]
    }
}
```

---

#### 4.2 Stakeholder-Specific Views

**Module:** `src/insight_generator.py` → `InsightGenerator.create_stakeholder_view()`

**What it does:**
- Filters and formats insights for different audiences:
  - **Product:** Feature implications, user impact
  - **Design:** Pain points, design opportunities
  - **Leadership:** Risks, opportunities, trends
  - **Research:** Full methodological detail

**Why UXRs need it:**
- Prevents reworking outputs for different audiences
- Increases perceived value of research
- Speaks each stakeholder's language

**Example (Product View):**
```python
{
    'stakeholder': 'Product Management',
    'focus': 'Feature-level implications and user impact',
    'insights': [
        {
            'headline': 'Users frustrated with automation',
            'impact': 'Critical',
            'affected_users': '35%',
            'feature_implications': ['Address automation concerns', 'Prioritize improvements'],
            'action_required': True
        }
    ]
}
```

---

#### 4.3 Analyst Annotation Layer

**Module:** `src/insight_generator.py` → `InsightGenerator.add_analyst_annotation()`

**What it does:**
- Lets UXRs rename, merge, split themes
- Attach analytic memos
- Track what was model-derived vs. analyst-shaped

**Why UXRs need it:**
- Preserves qualitative epistemology
- Prevents deskilling
- Maintains analyst agency

**Example:**
```python
{
    'type': 'rename',
    'content': 'Automation Anxiety',
    'analyst': 'Jane Doe',
    'timestamp': '2026-02-01T15:30:00',
    'original_insight': {...}
}
```

---

#### 4.4 Longitudinal Pattern Detection

**Module:** `src/insight_generator.py` → `InsightGenerator.detect_longitudinal_patterns()`

**What it does:**
- Compares themes across time periods
- Identifies emerging, declining, and persistent themes
- Detects sentiment shifts over time

**Why UXRs need it:**
- Industry loves "trend" narratives
- Shows evolution of user needs
- Justifies ongoing research investment

**Example output:**
```python
{
    'emerging_themes': [
        {
            'theme': 'AI integration concerns',
            'status': 'New theme',
            'current_prevalence': 0.18
        }
    ],
    'declining_themes': [
        {
            'theme': 'Mobile app performance',
            'decline': 45.0,  # 45% decrease
            'status': 'Declining'
        }
    ]
}
```

---

## 🎨 Visualizations

All features include publication-ready visualizations:

1. **Theme Stability:** Bar charts with robustness color-coding
2. **Theme Overlap:** Heatmaps with fuzziness indicators
3. **Salience vs. Emotional Weight:** Quadrant scatterplots
4. **Contrastive Analysis:** Side-by-side bar charts
5. **Theme × Sentiment:** Grouped bar charts with polarity indicators
6. **Participation Imbalance:** Distribution bars + Lorenz curves
7. **Emergent vs. Dominant:** Frequency-intensity scatterplots

**Module:** `src/advanced_visualizations.py`

---

## 📖 Usage Examples

### Example 1: Complete UX Research Workflow

```python
from src.theme_analysis import ThemeAnalyzer
from src.contrastive_analysis import ContrastiveAnalyzer
from src.quote_analysis import QuoteAnalyzer
from src.insight_generator import InsightGenerator
from src.advanced_visualizations import AdvancedVisualizer

# 1. Assess theme stability
theme_analyzer = ThemeAnalyzer()
stability = theme_analyzer.compute_theme_stability(
    texts, vectorizer, topic_model, n_resamples=10
)

# 2. Identify emergent themes
emergent = theme_analyzer.identify_emergent_themes(
    doc_topic_matrix, sentiment_scores
)

# 3. Compare groups
contrastive = ContrastiveAnalyzer()
comparison = contrastive.generate_contrastive_summary(
    texts, groups, doc_topic_matrix, topic_terms, 'SMEs', 'Enterprises'
)

# 4. Create provenance chains
quote_analyzer = QuoteAnalyzer()
provenance = quote_analyzer.create_provenance_chain(
    texts, doc_topic_matrix, topic_id=0, metadata=df
)

# 5. Generate insights
insight_gen = InsightGenerator()
insight = insight_gen.generate_insight_statement(
    topic_id=0,
    topic_terms=['automation', 'workflow'],
    prevalence=0.35,
    sentiment=-0.32,
    n_participants=12
)

# 6. Create stakeholder view
product_view = insight_gen.create_stakeholder_view(
    [insight], stakeholder_type='product'
)

# 7. Visualize
viz = AdvancedVisualizer()
fig = viz.plot_salience_emotional_weight(salience_data)
```

---

## 🔬 Methodological Notes

### Confidence Assessment
Confidence is based on:
- **Prevalence:** Higher = more confident
- **Stability:** More robust themes = more confident
- **Participant count:** More contributors = more confident

### Priority Assessment
Priority is based on:
- **Prevalence:** Higher = higher priority
- **Sentiment:** Negative = higher priority
- **Group differences:** Present = higher priority

### Limitations
- Insights are **model-derived** and should be validated with domain expertise
- Analyst judgment remains essential
- Tools augment, not replace, qualitative analysis skills

---

## 📚 References

These features are inspired by:
- UX research best practices
- Qualitative data analysis methodologies
- Industry needs for rigor + storytelling
- Concerns about AI deskilling researchers

---

## 🚀 Next Steps

1. **Integrate into Streamlit UI** - Make features accessible via web interface
2. **Create example notebooks** - Show real-world usage patterns
3. **Add export functionality** - Enable report generation
4. **Gather user feedback** - Iterate based on UXR needs

---

**Built with ❤️ for UX researchers who need both rigor and narrative power**