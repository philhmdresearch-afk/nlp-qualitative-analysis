# NLP Qualitative Data Analysis Tool

A comprehensive, energy-efficient tool for analyzing both **structured** (numeric/categorical) and **unstructured** (text) qualitative data. Built with Python and Streamlit for an intuitive web interface.

## 🌟 Features

### Unstructured Data (Text) Analysis
- **Text Preprocessing**: Cleaning, tokenization, lemmatization, stopword removal
- **Clustering**: K-means, Hierarchical, HDBSCAN, Spherical K-means
- **Topic Modeling**: LDA, NMF, LSA with coherence evaluation
- **Sentiment Analysis**: VADER-based sentiment scoring
- **Statistical Tests**: Chi-square tests for topic/cluster distributions
- **Visualizations**: Word clouds, dendrograms, distribution charts

### 🎓 Advanced UX Research Features (NEW!)
- **Theme Stability Analysis**: Bootstrap resampling to assess theme robustness
- **Theme Overlap Detection**: Identify boundary documents and theme fuzziness
- **Emergent vs. Dominant Themes**: Separate weak signals from high-frequency themes
- **Contrastive Analysis**: Natural language summaries comparing groups
- **Theme × Sentiment Interactions**: Detect polarity shifts across segments
- **Salience vs. Emotional Weight**: Prioritization matrix for themes
- **Quote Provenance**: Full audit trails linking themes to quotes with metadata
- **Representativeness Indicators**: Label quotes as representative/edge cases
- **Participation Imbalance Detection**: Identify dominant voices and bias
- **Question Bias Detection**: Analyze if questions drive specific themes/sentiment
- **Insight Generation**: Auto-generate short, memorable insights (Context + Problem → Action)
- **Stakeholder Views**: Tailored outputs for Product, Design, Leadership, Research
- **Analyst Annotations**: Preserve qualitative epistemology with override layer
- **Longitudinal Patterns**: Track theme evolution over time

📖 **[See full UX Research Features documentation](docs/UX_RESEARCH_FEATURES.md)**

### 📝 Reflexive Thematic Analysis (RTA) Assistant (NEW!)

Complete implementation of Braun & Clarke's 6-phase RTA methodology with AI augmentation:

- **Phase 1: Familiarisation** - Dataset statistics and reading guidance
- **Phase 2: Initial Coding** - AI candidate codes as provocations (not ground truth)
- **Phase 3: Searching for Themes** - Theme construction with central organizing concepts
- **Phase 4: Reviewing Themes** - Coherence checking and refinement
- **Phase 5: Defining and Naming** - Theme essence and boundaries
- **Phase 6: Producing the Report** - Methods section with AI transparency

**Key Features:**
- ✅ Reflexive memo system (primary analytic tool)
- ✅ Complete audit trail of all decisions
- ✅ Positionality statement tracking
- ✅ Human-in-the-loop approach (AI augments, never replaces)
- ✅ Epistemology/ontology documentation
- ✅ Export to JSON/Markdown

📖 **[See RTA Assistant Guide](docs/RTA_ASSISTANT_GUIDE.md)**

### Structured Data Analysis
- **Preprocessing**: Scaling, encoding, missing value handling
- **Clustering**: K-means, Gaussian Mixture, Hierarchical, K-modes/K-prototypes
- **Distance Metrics**: Euclidean, Manhattan, Gower (for mixed types)
- **Dimensionality Reduction**: PCA, UMAP, t-SNE
- **Statistical Tests**: Chi-square, Mann-Whitney, Kruskal-Wallis
- **Visualizations**: Scatter plots, heatmaps, dendrograms

## 📋 Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for large datasets)
- ~400MB disk space for dependencies

## 🚀 Quick Start

### Option 1: Use in Browser (Recommended)

**Deploy to Streamlit Cloud (Free):**

1. Fork this repository on GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app" and select this repository
5. Set main file: `app.py`
6. Click "Deploy!"

Your app will be live at: `https://YOUR_USERNAME-nlp-qualitative-analysis.streamlit.app`

📖 **[See full deployment guide](DEPLOYMENT.md)**

### Option 2: Run Locally

#### 1. Clone or Download

```bash
cd nlp-qualitative-analysis
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (First Time Only)

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

### 5. Download spaCy Model (First Time Only)

```bash
python -m spacy download en_core_web_sm
```

## 🎯 Quick Start

### Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Select Data Type**: Choose "Unstructured (Text)" or "Structured (Numeric/Categorical)"
2. **Upload File**: Supported formats: CSV, Excel, TXT, JSON
3. **Configure Analysis**: Select preprocessing options and analysis parameters
4. **Run Analysis**: Click buttons to perform clustering, topic modeling, sentiment analysis, etc.
5. **View Results**: Explore visualizations, statistics, and export results

## 📊 Example Usage

### Unstructured Text Analysis

```python
from src.data_loader import DataLoader
from src.text_preprocessor import TextPreprocessor
from src.clustering import ClusterAnalyzer
from src.topic_modeling import TopicModeler
from src.sentiment_analysis import SentimentAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
loader = DataLoader()
df = loader.load_data('survey_responses.csv', data_type='unstructured')
texts = df['response'].tolist()

# Preprocess
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_stopwords=True,
    lemmatize=True
)
processed_texts = preprocessor.preprocess_batch(texts)

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(processed_texts)

# Cluster
clusterer = ClusterAnalyzer(algorithm='kmeans', n_clusters=5)
clusterer.fit(X.toarray())
print(f"Silhouette Score: {clusterer.metrics_['silhouette']:.3f}")

# Topic Modeling
modeler = TopicModeler(method='lda', n_topics=5)
modeler.fit(processed_texts)
topics = modeler.get_top_terms(n_terms=10)

# Sentiment Analysis
analyzer = SentimentAnalyzer()
sentiments = analyzer.analyze_batch(texts)
print(f"Average Sentiment: {sentiments['compound'].mean():.3f}")
```

### Structured Data Analysis

```python
from src.data_loader import DataLoader
from src.structured_preprocessor import StructuredPreprocessor
from src.clustering import ClusterAnalyzer

# Load data
loader = DataLoader()
df = loader.load_data('survey_data.csv', data_type='structured')

# Preprocess
preprocessor = StructuredPreprocessor(
    numeric_strategy='standard',
    categorical_strategy='onehot'
)
X = preprocessor.fit_transform(df)

# Cluster
clusterer = ClusterAnalyzer(algorithm='kmeans', n_clusters=3)
clusterer.fit(X)
print(f"Cluster sizes: {clusterer.metrics_['cluster_sizes']}")
```

## 📁 Project Structure

```
nlp-qualitative-analysis/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py             # Data loading for multiple formats
│   ├── text_preprocessor.py       # Text preprocessing (NLTK)
│   ├── structured_preprocessor.py # Structured data preprocessing
│   ├── clustering.py              # Clustering algorithms
│   ├── topic_modeling.py          # LDA, NMF, LSA
│   ├── sentiment_analysis.py      # VADER sentiment analysis
│   ├── statistical_tests.py       # Chi-square, Mann-Whitney, etc.
│   └── visualization.py           # Charts, word clouds, dendrograms
├── data/                          # Sample data (optional)
├── notebooks/                     # Jupyter notebooks (optional)
├── tests/                         # Unit tests (optional)
└── docs/                          # Additional documentation
```

## 🔧 Configuration Options

### Text Preprocessing
- `lowercase`: Convert to lowercase (default: True)
- `remove_punctuation`: Remove punctuation marks (default: True)
- `remove_stopwords`: Remove common stopwords (default: True)
- `lemmatize`: Apply lemmatization (default: True)
- `min_word_length`: Minimum word length (default: 2)

### Clustering
- `algorithm`: 'kmeans', 'hierarchical', 'hdbscan', 'gmm', 'spherical_kmeans'
- `n_clusters`: Number of clusters (2-20)
- `random_state`: Random seed for reproducibility

### Topic Modeling
- `method`: 'lda', 'nmf', 'lsa'
- `n_topics`: Number of topics (2-20)
- `max_features`: Maximum vocabulary size (100-5000)

### Sentiment Analysis
- Uses VADER (no configuration needed)
- Returns: compound, positive, neutral, negative scores

## 📈 Evaluation Metrics

### Clustering Metrics
- **Silhouette Score**: Measures cluster cohesion (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **Calinski-Harabasz Score**: Ratio of between/within cluster variance (higher is better)

### Topic Modeling Metrics
- **Coherence Score**: Measures topic interpretability (higher is better)
- **Perplexity**: LDA model fit (lower is better)

### Statistical Tests
- **Chi-Square Test**: Tests independence between categorical variables
- **Cramér's V**: Effect size for chi-square (0 to 1)
- **Mann-Whitney U**: Non-parametric test for two groups
- **Kruskal-Wallis**: Non-parametric test for multiple groups

## 🎨 Visualizations

### Available Visualizations
1. **Bar Charts**: Distribution of clusters, topics, sentiments
2. **Dendrograms**: Hierarchical clustering relationships
3. **Word Clouds**: Visual representation of term frequencies
4. **Scatter Plots**: 2D/3D cluster visualizations
5. **Heatmaps**: Correlation matrices, contingency tables
6. **Sentiment Distributions**: Histogram and pie charts

## 📝 Supported Data Formats

### Input Formats
- **CSV**: Comma-separated values
- **Excel**: .xlsx, .xls files
- **JSON**: Structured JSON data
- **TXT**: Plain text (one document per line)

### Output Formats
- **CSV**: Results tables
- **PNG/PDF**: Visualizations
- **JSON**: Analysis results
- **HTML**: Interactive reports

## ⚡ Energy Efficiency Features

This tool is designed with energy efficiency in mind:

1. **Lazy Loading**: Libraries loaded only when needed
2. **Sparse Matrices**: Memory-efficient TF-IDF representation
3. **Batch Processing**: Process data in chunks
4. **Caching**: Avoid redundant computations
5. **Vectorized Operations**: NumPy/SciPy optimizations
6. **No GPU Required**: All CPU-based algorithms
7. **Minimal Dependencies**: ~400MB vs 2-3GB for transformer-based tools

## 📜 License Information

All dependencies use permissive open-source licenses:

- **pandas, numpy, scikit-learn, scipy**: BSD-3 License
- **NLTK**: Apache 2.0 License
- **Streamlit, plotly**: Apache 2.0 License
- **matplotlib**: PSF License (permissive)
- **wordcloud**: MIT License
- **gower, hdbscan, kmodes, umap-learn**: BSD-3 or MIT License

✅ **All licenses allow commercial use without restrictions**

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

**2. NLTK Data Missing**
```python
import nltk
nltk.download('all')  # Download all NLTK data
```

**3. Memory Errors**
- Reduce `max_features` parameter
- Process data in smaller batches
- Use `minibatch_kmeans` instead of `kmeans`

**4. Slow Performance**
- Disable lemmatization for faster preprocessing
- Reduce number of clusters/topics
- Use smaller `max_features` value

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional clustering algorithms
- More topic modeling methods
- Enhanced visualizations
- Performance optimizations
- Additional statistical tests

## 📧 Support

For issues, questions, or suggestions:
1. Check the documentation
2. Review example notebooks
3. Open an issue on GitHub

## 🎓 Citation

If you use this tool in your research, please cite:

```bibtex
@software{nlp_qualitative_analysis,
  title = {NLP Qualitative Data Analysis Tool},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/nlp-qualitative-analysis}
}
```

## 🔮 Future Enhancements

Planned features:
- [ ] Transformer-based embeddings (optional)
- [ ] Multi-language support
- [ ] Real-time analysis
- [ ] API endpoints
- [ ] Docker deployment
- [ ] Cloud hosting options
- [ ] Advanced topic modeling (BERTopic, Top2Vec)
- [ ] Network analysis for text
- [ ] Time series analysis

---

**Built with ❤️ for qualitative researchers**