"""
Streamlit Web Application for NLP Qualitative Data Analysis
Main interface for structured and unstructured data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from text_preprocessor import TextPreprocessor, create_default_preprocessor
from structured_preprocessor import StructuredPreprocessor
from clustering import ClusterAnalyzer, compute_hierarchical_linkage
from topic_modeling import TopicModeler
from sentiment_analysis import SentimentAnalyzer
from statistical_tests import StatisticalTester, create_contingency_table
from visualization import Visualizer
from theme_analysis import ThemeAnalyzer
from contrastive_analysis import ContrastiveAnalyzer
from quote_analysis import QuoteAnalyzer
from insight_generator import InsightGenerator
from advanced_visualizations import AdvancedVisualizer
from rta_assistant import RTAAssistant

# Page configuration
st.set_page_config(
    page_title="NLP Qualitative Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">📊 NLP Qualitative Data Analysis Tool</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Welcome!</b> This tool provides comprehensive analysis for both structured and unstructured data:
    <ul>
        <li><b>Structured Data:</b> Numeric/categorical clustering, statistical tests, visualizations</li>
        <li><b>Unstructured Data:</b> Text clustering, topic modeling, sentiment analysis, word clouds</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data type selection
        data_type = st.radio(
            "Select Data Type",
            ["Unstructured (Text)", "Structured (Numeric/Categorical)"],
            help="Choose the type of data you want to analyze"
        )
        
        st.markdown("---")
        
        # File upload
        st.subheader("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'txt', 'json'],
            help="Supported formats: CSV, Excel, TXT, JSON"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Main content
    if uploaded_file is None:
        st.info("👈 Please upload a data file to begin analysis")
        
        # Show example
        with st.expander("📖 View Example Data Formats"):
            st.markdown("""
            **Unstructured Data (Text):**
            - CSV with a text column
            - Plain text file (one document per line)
            - JSON with text field
            
            **Structured Data:**
            - CSV/Excel with numeric and/or categorical columns
            - Each row is an observation
            - Columns are features
            """)
        
        return
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            loader = DataLoader()
            
            # Save uploaded file temporarily
            temp_path = Path("temp_upload") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Load data
            df = loader.load_data(
                temp_path,
                data_type='unstructured' if 'Text' in data_type else 'structured'
            )
            
            st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Show data preview
    with st.expander("👀 Preview Data"):
        st.dataframe(df.head(10))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Analysis based on data type
    if "Text" in data_type:
        run_unstructured_analysis(df, loader)
    else:
        run_structured_analysis(df, loader)


def run_unstructured_analysis(df: pd.DataFrame, loader: DataLoader):
    """Run analysis for unstructured text data."""
    
    st.markdown('<div class="sub-header">📝 Unstructured Data Analysis</div>', unsafe_allow_html=True)
    
    # Column selection
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_columns:
        st.error("No text columns found in the data")
        return
    
    text_column = st.selectbox("Select text column", text_columns)
    texts = df[text_column].astype(str).tolist()
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 Preprocessing",
        "🎯 Clustering",
        "📚 Topic Modeling",
        "😊 Sentiment Analysis",
        "📊 Statistical Tests",
        "🎨 Advanced UX Features",
        "📝 RTA Assistant"
    ])
    
    # Tab 1: Preprocessing
    with tab1:
        st.subheader("Text Preprocessing")
        
        col1, col2 = st.columns(2)
        with col1:
            remove_stopwords = st.checkbox("Remove stopwords", value=True)
            lemmatize = st.checkbox("Lemmatize", value=True)
        with col2:
            remove_punctuation = st.checkbox("Remove punctuation", value=True)
            lowercase = st.checkbox("Lowercase", value=True)
        
        if st.button("Preprocess Texts", type="primary"):
            with st.spinner("Preprocessing..."):
                preprocessor = TextPreprocessor(
                    lowercase=lowercase,
                    remove_punctuation=remove_punctuation,
                    remove_stopwords=remove_stopwords,
                    lemmatize=lemmatize
                )
                
                processed_texts = preprocessor.preprocess_batch(texts)
                st.session_state['processed_texts'] = processed_texts
                
                # Show statistics
                stats = preprocessor.get_statistics(processed_texts)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Documents", stats['n_documents'])
                col2.metric("Vocabulary Size", stats['vocabulary_size'])
                col3.metric("Avg Length", f"{stats['avg_doc_length']:.1f}")
                col4.metric("Total Tokens", stats['total_tokens'])
                
                # Show sample
                st.write("**Sample Processed Text:**")
                st.text(processed_texts[0][:500] + "...")
                
                # Top words
                st.write("**Top 10 Words:**")
                top_words_df = pd.DataFrame(stats['top_10_words'], columns=['Word', 'Frequency'])
                st.dataframe(top_words_df)
    
    # Tab 2: Clustering
    with tab2:
        st.subheader("Text Clustering")
        
        if 'processed_texts' not in st.session_state:
            st.warning("⚠️ Please preprocess texts first")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
            with col2:
                algorithm = st.selectbox("Algorithm", ['kmeans', 'hierarchical', 'hdbscan'])
            with col3:
                max_features = st.number_input("Max features", 100, 5000, 1000)
            
            if st.button("Run Clustering", type="primary"):
                with st.spinner("Clustering..."):
                    # Vectorize
                    vectorizer = TfidfVectorizer(max_features=max_features)
                    X = vectorizer.fit_transform(st.session_state['processed_texts'])
                    
                    # Cluster
                    clusterer = ClusterAnalyzer(
                        algorithm=algorithm,
                        n_clusters=n_clusters
                    )
                    clusterer.fit(X.toarray())
                    
                    st.session_state['cluster_labels'] = clusterer.labels_
                    st.session_state['cluster_model'] = clusterer
                    
                    # Show metrics
                    st.write("**Clustering Metrics:**")
                    col1, col2, col3 = st.columns(3)
                    if clusterer.metrics_.get('silhouette'):
                        col1.metric("Silhouette Score", f"{clusterer.metrics_['silhouette']:.3f}")
                    if clusterer.metrics_.get('davies_bouldin'):
                        col2.metric("Davies-Bouldin", f"{clusterer.metrics_['davies_bouldin']:.3f}")
                    if clusterer.metrics_.get('calinski_harabasz'):
                        col3.metric("Calinski-Harabasz", f"{clusterer.metrics_['calinski_harabasz']:.1f}")
                    
                    # Cluster sizes
                    st.write("**Cluster Sizes:**")
                    cluster_sizes = pd.Series(clusterer.labels_).value_counts().sort_index()
                    st.bar_chart(cluster_sizes)
                    
                    # Word clouds per cluster
                    st.write("**Word Clouds by Cluster:**")
                    visualizer = Visualizer()
                    
                    for cluster_id in range(n_clusters):
                        mask = clusterer.labels_ == cluster_id
                        cluster_texts = ' '.join([st.session_state['processed_texts'][i] for i, m in enumerate(mask) if m])
                        
                        st.write(f"**Cluster {cluster_id}** ({mask.sum()} documents)")
                        fig = visualizer.create_wordcloud(cluster_texts, title=f"Cluster {cluster_id}")
                        st.pyplot(fig)
    
    # Tab 3: Topic Modeling
    with tab3:
        st.subheader("Topic Modeling")
        
        if 'processed_texts' not in st.session_state:
            st.warning("⚠️ Please preprocess texts first")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                n_topics = st.slider("Number of topics", 2, 10, 5)
            with col2:
                method = st.selectbox("Method", ['lda', 'nmf', 'lsa'])
            with col3:
                n_top_terms = st.slider("Top terms per topic", 5, 20, 10)
            
            if st.button("Run Topic Modeling", type="primary"):
                with st.spinner("Modeling topics..."):
                    modeler = TopicModeler(
                        method=method,
                        n_topics=n_topics
                    )
                    modeler.fit(st.session_state['processed_texts'])
                    
                    st.session_state['topic_model'] = modeler
                    
                    # Show topics
                    st.write("**Discovered Topics:**")
                    top_terms = modeler.get_top_terms(n_top_terms)
                    
                    for topic_id, terms_weights in top_terms.items():
                        with st.expander(f"📚 Topic {topic_id}"):
                            terms = [term for term, _ in terms_weights]
                            weights = [weight for _, weight in terms_weights]
                            
                            # Show terms
                            st.write("**Top Terms:**", ", ".join(terms[:10]))
                            
                            # Word cloud
                            visualizer = Visualizer()
                            term_freq = dict(terms_weights)
                            fig = visualizer.create_wordcloud(term_freq, title=f"Topic {topic_id}")
                            st.pyplot(fig)
                    
                    # Topic distribution
                    topic_summary = modeler.get_topic_summary(n_top_terms)
                    st.write("**Topic Summary:**")
                    st.dataframe(topic_summary[['topic_id', 'top_terms', 'n_documents']])
    
    # Tab 4: Sentiment Analysis
    with tab4:
        st.subheader("Sentiment Analysis")
        
        if st.button("Analyze Sentiment", type="primary"):
            with st.spinner("Analyzing sentiment..."):
                analyzer = SentimentAnalyzer()
                sentiments = analyzer.analyze_batch(texts)
                
                st.session_state['sentiments'] = sentiments
                
                # Show statistics
                stats = analyzer.get_statistics(texts)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Compound", f"{stats['avg_compound']:.3f}")
                col2.metric("Positive %", f"{stats['positive_pct']:.1f}%")
                col3.metric("Neutral %", f"{stats['neutral_pct']:.1f}%")
                col4.metric("Negative %", f"{stats['negative_pct']:.1f}%")
                
                # Visualizations
                visualizer = Visualizer()
                fig = visualizer.plot_sentiment_distribution(sentiments)
                st.pyplot(fig)
                
                # Most positive/negative
                extremes = analyzer.get_most_positive_negative(texts, n=3)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Most Positive:**")
                    for text, score in extremes['most_positive']:
                        st.success(f"Score: {score:.3f}")
                        st.write(text[:200] + "...")
                
                with col2:
                    st.write("**Most Negative:**")
                    for text, score in extremes['most_negative']:
                        st.error(f"Score: {score:.3f}")
                        st.write(text[:200] + "...")
    
    # Tab 5: Statistical Tests
    with tab5:
        st.subheader("Statistical Tests")
        
        if 'cluster_labels' in st.session_state or 'topic_model' in st.session_state:
            st.write("**Chi-Square Test for Distribution Differences**")
            
            # Group selection
            group_columns = [col for col in df.columns if col != text_column]
            if group_columns:
                group_col = st.selectbox("Select grouping variable", group_columns)
                
                if st.button("Run Chi-Square Test", type="primary"):
                    tester = StatisticalTester()
                    
                    if 'cluster_labels' in st.session_state:
                        results = tester.compare_cluster_distributions(
                            st.session_state['cluster_labels'],
                            df[group_col].values
                        )
                        
                        st.write("**Test Results:**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Chi-Square", f"{results['statistic']:.2f}")
                        col2.metric("P-value", f"{results['p_value']:.4f}")
                        col3.metric("Cramér's V", f"{results['cramers_v']:.3f}")
                        
                        if results['significant']:
                            st.success("✅ Significant difference detected!")
                        else:
                            st.info("ℹ️ No significant difference detected")
                        
                        # Contingency table
                        st.write("**Contingency Table:**")
                        contingency_df = pd.DataFrame(
                            results['contingency_table'],
                            index=[f"Cluster {i}" for i in results['clusters']],
                            columns=[f"Group {g}" for g in results['groups']]
                        )
                        st.dataframe(contingency_df)
            else:
                st.info("No grouping variables available for comparison")
        else:
            st.warning("⚠️ Please run clustering or topic modeling first")
    
    # Tab 6: Advanced UX Features
    with tab6:
        st.subheader("Advanced UX Research Features")
        
        st.markdown("""
        <div class="info-box">
        <b>Advanced Features for UX Research:</b>
        <ul>
            <li><b>Theme Analysis:</b> Stability, coherence, overlap detection</li>
            <li><b>Contrastive Analysis:</b> Compare groups, sentiment interactions</li>
            <li><b>Quote Analysis:</b> Provenance tracking, representativeness</li>
            <li><b>Insight Generation:</b> Narrative-ready insights for stakeholders</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if 'processed_texts' not in st.session_state or 'cluster_labels' not in st.session_state:
            st.warning("⚠️ Please preprocess texts and run clustering first")
        else:
            feature_type = st.selectbox(
                "Select Feature",
                ["Theme Stability", "Theme Overlap", "Contrastive Analysis", "Quote Provenance", "Generate Insights"]
            )
            
            if feature_type == "Theme Stability":
                st.write("**Theme Stability Analysis**")
                st.info("Assess how robust your themes are using bootstrap resampling")
                
                if st.button("Analyze Theme Stability", type="primary"):
                    with st.spinner("Analyzing theme stability..."):
                        theme_analyzer = ThemeAnalyzer()
                        
                        # Create theme assignments from clusters
                        theme_assignments = {
                            f"theme_{i}": [j for j, label in enumerate(st.session_state['cluster_labels']) if label == i]
                            for i in range(max(st.session_state['cluster_labels']) + 1)
                        }
                        
                        stability = theme_analyzer.assess_theme_stability(
                            st.session_state['processed_texts'],
                            theme_assignments,
                            n_bootstrap=100
                        )
                        
                        st.write("**Stability Scores:**")
                        for theme, score in stability['stability_scores'].items():
                            col1, col2 = st.columns([3, 1])
                            col1.write(f"**{theme}**")
                            col2.metric("Stability", f"{score:.2%}")
                        
                        st.write(f"**Overall Stability:** {stability['overall_stability']:.2%}")
            
            elif feature_type == "Theme Overlap":
                st.write("**Theme Overlap Detection**")
                st.info("Identify documents that span multiple themes")
                
                if st.button("Detect Overlaps", type="primary"):
                    with st.spinner("Detecting overlaps..."):
                        theme_analyzer = ThemeAnalyzer()
                        
                        # Create theme assignments
                        theme_assignments = {
                            f"theme_{i}": [j for j, label in enumerate(st.session_state['cluster_labels']) if label == i]
                            for i in range(max(st.session_state['cluster_labels']) + 1)
                        }
                        
                        overlaps = theme_analyzer.detect_theme_overlap(
                            st.session_state['processed_texts'],
                            theme_assignments
                        )
                        
                        st.write(f"**Overlapping Documents:** {len(overlaps['overlapping_docs'])}")
                        
                        if overlaps['overlapping_docs']:
                            st.write("**Sample Overlaps:**")
                            for doc_id in list(overlaps['overlapping_docs'])[:5]:
                                themes = overlaps['overlap_details'][doc_id]['themes']
                                st.write(f"- Document {doc_id}: {', '.join(themes)}")
            
            elif feature_type == "Contrastive Analysis":
                st.write("**Contrastive Analysis**")
                st.info("Compare themes across different groups")
                
                group_columns = [col for col in df.columns if col != text_column]
                if group_columns:
                    group_col = st.selectbox("Select grouping variable", group_columns)
                    
                    if st.button("Run Contrastive Analysis", type="primary"):
                        with st.spinner("Analyzing contrasts..."):
                            contrastive = ContrastiveAnalyzer()
                            
                            # Create theme assignments
                            theme_assignments = {
                                f"theme_{i}": [j for j, label in enumerate(st.session_state['cluster_labels']) if label == i]
                                for i in range(max(st.session_state['cluster_labels']) + 1)
                            }
                            
                            comparison = contrastive.compare_groups(
                                st.session_state['processed_texts'],
                                theme_assignments,
                                df[group_col].values
                            )
                            
                            st.write("**Group Comparison:**")
                            for theme, data in comparison['theme_distributions'].items():
                                st.write(f"**{theme}**")
                                st.bar_chart(data)
                else:
                    st.warning("No grouping variables available")
            
            elif feature_type == "Quote Provenance":
                st.write("**Quote Provenance Tracking**")
                st.info("Track the origin and context of quotes")
                
                if st.button("Analyze Quotes", type="primary"):
                    with st.spinner("Analyzing quotes..."):
                        quote_analyzer = QuoteAnalyzer()
                        
                        # Analyze first few documents
                        sample_texts = st.session_state['processed_texts'][:10]
                        
                        provenance = quote_analyzer.create_provenance_chain(
                            sample_texts,
                            list(range(len(sample_texts)))
                        )
                        
                        st.write(f"**Analyzed {len(provenance)} documents**")
                        st.write("**Sample Provenance:**")
                        for doc_id, chain in list(provenance.items())[:3]:
                            st.write(f"- Document {doc_id}: {len(chain['transformations'])} transformations")
            
            elif feature_type == "Generate Insights":
                st.write("**Generate Narrative Insights**")
                st.info("Create stakeholder-ready insight statements")
                
                stakeholder = st.selectbox(
                    "Target Stakeholder",
                    ["Executive", "Product Manager", "Designer", "Engineer"]
                )
                
                if st.button("Generate Insights", type="primary"):
                    with st.spinner("Generating insights..."):
                        insight_gen = InsightGenerator()
                        
                        # Create theme data
                        theme_data = {
                            f"theme_{i}": {
                                'documents': [j for j, label in enumerate(st.session_state['cluster_labels']) if label == i],
                                'keywords': [f"keyword_{i}_{k}" for k in range(5)]
                            }
                            for i in range(max(st.session_state['cluster_labels']) + 1)
                        }
                        
                        insights = insight_gen.generate_insights(
                            st.session_state['processed_texts'],
                            theme_data,
                            stakeholder_type=stakeholder.lower().replace(' ', '_')
                        )
                        
                        st.write("**Generated Insights:**")
                        for insight in insights[:5]:
                            with st.expander(f"💡 {insight['title']}"):
                                st.write(f"**Context:** {insight['context']}")
                                st.write(f"**Problem:** {insight['problem']}")
                                st.write(f"**Action:** {insight['action']}")
                                st.write(f"**Impact:** {insight['impact']}")
    
    # Tab 7: RTA Assistant
    with tab7:
        st.subheader("Reflexive Thematic Analysis (RTA) Assistant")
        
        st.markdown("""
        <div class="info-box">
        <b>RTA Assistant - Braun & Clarke Methodology:</b><br>
        This tool guides you through all 6 phases of Reflexive Thematic Analysis while maintaining
        your analytic agency. AI suggestions are provocations, not ground truth.
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize RTA session
        if 'rta_session' not in st.session_state:
            st.session_state['rta_session'] = None
        
        # Project setup
        with st.expander("🎯 Project Setup", expanded=(st.session_state['rta_session'] is None)):
            project_name = st.text_input("Project Name", "My RTA Project")
            researcher_name = st.text_input("Your Name", "Researcher")
            
            col1, col2 = st.columns(2)
            with col1:
                epistemology = st.selectbox(
                    "Epistemology",
                    ["contextualist", "constructionist", "realist", "relativist"]
                )
            with col2:
                ontology = st.selectbox(
                    "Ontology",
                    ["critical realist", "relativist", "realist", "constructionist"]
                )
            
            positionality = st.text_area(
                "Positionality Statement (REQUIRED)",
                "Describe your background, assumptions, and how they shape your analysis...",
                height=150
            )
            
            if st.button("Initialize RTA Project", type="primary"):
                rta = RTAAssistant(
                    project_name=project_name,
                    researcher_name=researcher_name,
                    epistemology=epistemology,
                    ontology=ontology
                )
                rta.set_positionality_statement(positionality)
                st.session_state['rta_session'] = rta
                st.success("✅ RTA project initialized!")
        
        if st.session_state['rta_session'] is not None:
            rta = st.session_state['rta_session']
            
            # Phase selection
            phase = st.selectbox(
                "Select RTA Phase",
                [
                    "Phase 1: Familiarisation",
                    "Phase 2: Generating Initial Codes",
                    "Phase 3: Searching for Themes",
                    "Phase 4: Reviewing Themes",
                    "Phase 5: Defining and Naming Themes",
                    "Phase 6: Producing the Report"
                ]
            )
            
            if "Phase 1" in phase:
                st.write("**Phase 1: Familiarisation with the Data**")
                
                if st.button("Start Phase 1", type="primary"):
                    with st.spinner("Analyzing data..."):
                        phase1 = rta.phase1_familiarisation(texts)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Documents", phase1['statistics']['n_documents'])
                        col2.metric("Total Words", phase1['statistics']['total_words'])
                        col3.metric("Avg Length", f"{phase1['statistics']['avg_length']:.1f}")
                        
                        st.write("**Your Task:**")
                        for instruction in phase1['guidance']['instructions']:
                            st.write(f"- {instruction}")
                        
                        st.warning(phase1['guidance']['warning'])
                
                memo = st.text_area("Add Reflexive Memo", height=150)
                if st.button("Save Memo"):
                    rta.add_reflexive_memo(memo, memo_type="general")
                    st.success("✅ Memo saved")
            
            elif "Phase 2" in phase:
                st.write("**Phase 2: Generating Initial Codes**")
                
                request_ai = st.checkbox("Request AI candidate codes (as provocations)", value=True)
                
                if st.button("Start Phase 2", type="primary"):
                    with st.spinner("Generating codes..."):
                        phase2 = rta.phase2_initial_coding(texts, request_ai_suggestions=request_ai)
                        
                        if 'ai_suggestions' in phase2:
                            st.write("**AI Candidate Codes (PROVOCATIONS):**")
                            for code in phase2['ai_suggestions']['candidate_codes'][:10]:
                                st.write(f"- **{code['suggested_code']}**: {code['related_terms']}")
                            
                            st.warning(phase2['ai_note'])
                
                col1, col2 = st.columns(2)
                with col1:
                    code_name = st.text_input("Code Name")
                with col2:
                    code_desc = st.text_input("Code Description")
                
                if st.button("Add Code"):
                    rta.add_code(code_name, code_desc, ["example"], ai_suggested=False)
                    st.success(f"✅ Code '{code_name}' added")
            
            elif "Phase 3" in phase:
                st.write("**Phase 3: Searching for Themes**")
                
                st.info(f"You have {len(rta.codes)} codes to organize into themes")
                
                col1, col2 = st.columns(2)
                with col1:
                    theme_name = st.text_input("Theme Name")
                with col2:
                    central_concept = st.text_input("Central Organizing Concept")
                
                description = st.text_area("Theme Description")
                rationale = st.text_area("Conceptual Rationale")
                
                if st.button("Create Theme"):
                    rta.add_theme(
                        theme_name=theme_name,
                        central_concept=central_concept,
                        included_codes=list(rta.codes.keys()),
                        description=description,
                        conceptual_rationale=rationale
                    )
                    st.success(f"✅ Theme '{theme_name}' created")
            
            # Project summary
            st.markdown("---")
            summary = rta.get_project_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Codes", summary['n_codes'])
            col2.metric("Themes", summary['n_themes'])
            col3.metric("Memos", summary['n_memos'])
            col4.metric("AI Uses", summary['n_ai_assistance_instances'])
            
            # Export options
            st.markdown("---")
            st.write("**Export Options:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Methods Section"):
                    methods = rta.generate_methods_section()
                    st.text_area("Methods Section", methods, height=400)
            
            with col2:
                export_format = st.selectbox("Export Format", ["JSON", "Markdown"])
                if st.button("Export Audit Trail"):
                    audit = rta.export_audit_trail(format=export_format.lower())
                    st.download_button(
                        "Download Audit Trail",
                        audit,
                        file_name=f"rta_audit.{export_format.lower()}",
                        mime="text/plain"
                    )


def run_structured_analysis(df: pd.DataFrame, loader: DataLoader):
    """Run analysis for structured data."""
    
    st.markdown('<div class="sub-header">📊 Structured Data Analysis</div>', unsafe_allow_html=True)
    
    st.info("🚧 Structured data analysis interface coming soon! The backend modules are ready.")
    
    # Show data types
    st.write("**Column Types:**")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null': df.count(),
        'Unique': df.nunique()
    })
    st.dataframe(dtypes_df)


if __name__ == "__main__":
    main()

# Made with Bob
