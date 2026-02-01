# Quick Start Guide

Get up and running with the NLP Qualitative Analysis Tool in 5 minutes!

## 🚀 Installation (5 minutes)

### Step 1: Install Python
Ensure you have Python 3.8+ installed:
```bash
python --version
```

### Step 2: Install Dependencies
```bash
cd nlp-qualitative-analysis
pip install -r requirements.txt
```

### Step 3: Download Required Data
```bash
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🎯 Run the Application

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

## 📝 First Analysis - Text Data

### 1. Prepare Your Data

Create a CSV file with a text column:

```csv
id,text,category
1,"I love this product! It's amazing and works perfectly.",positive
2,"Terrible experience. Would not recommend to anyone.",negative
3,"It's okay, nothing special but does the job.",neutral
```

### 2. Upload and Analyze

1. **Select Data Type**: Choose "Unstructured (Text)"
2. **Upload File**: Click "Browse files" and select your CSV
3. **Select Text Column**: Choose the column containing your text
4. **Preprocess**: 
   - Check "Remove stopwords"
   - Check "Lemmatize"
   - Click "Preprocess Texts"
5. **Run Clustering**:
   - Set number of clusters (e.g., 3)
   - Click "Run Clustering"
   - View word clouds for each cluster
6. **Analyze Sentiment**:
   - Click "Analyze Sentiment"
   - View sentiment distribution
7. **Statistical Tests**:
   - Select grouping variable (e.g., "category")
   - Click "Run Chi-Square Test"
   - View significance results

## 📊 First Analysis - Structured Data

### 1. Prepare Your Data

Create a CSV file with numeric/categorical columns:

```csv
age,income,education,satisfaction
25,50000,Bachelor,4
35,75000,Master,5
45,60000,Bachelor,3
```

### 2. Upload and Analyze

1. **Select Data Type**: Choose "Structured (Numeric/Categorical)"
2. **Upload File**: Click "Browse files" and select your CSV
3. **View Data**: Check the data preview
4. **Configure Analysis**: (Coming soon in the interface)

## 💡 Tips for Best Results

### Text Data
- **Preprocessing**: Always preprocess before clustering/topic modeling
- **Cluster Count**: Start with 3-5 clusters, adjust based on results
- **Topic Count**: Use 3-7 topics for most datasets
- **Text Length**: Works best with 50-500 words per document

### Structured Data
- **Missing Values**: Handle missing data before analysis
- **Scaling**: Use standard scaling for most numeric data
- **Encoding**: Use one-hot encoding for categorical variables
- **Cluster Count**: Use elbow method to find optimal k

## 🎨 Understanding Visualizations

### Word Clouds
- **Larger words** = more frequent/important
- **Colors** = different word groups
- **Best for**: Understanding cluster/topic themes

### Dendrograms
- **Height** = distance between clusters
- **Branches** = hierarchical relationships
- **Best for**: Understanding cluster structure

### Bar Charts
- **Height** = frequency/count
- **Colors** = different categories
- **Best for**: Comparing distributions

## 📈 Interpreting Results

### Clustering Metrics

**Silhouette Score** (-1 to 1)
- > 0.7: Strong clustering
- 0.5-0.7: Reasonable clustering
- 0.25-0.5: Weak clustering
- < 0.25: No meaningful clusters

**Davies-Bouldin Index** (lower is better)
- < 1.0: Good clustering
- 1.0-2.0: Acceptable
- > 2.0: Poor clustering

### Sentiment Scores

**Compound Score** (-1 to 1)
- ≥ 0.05: Positive
- -0.05 to 0.05: Neutral
- ≤ -0.05: Negative

### Statistical Tests

**P-value** (significance level: 0.05)
- < 0.05: Significant difference
- ≥ 0.05: No significant difference

**Cramér's V** (effect size)
- 0.0-0.1: Negligible
- 0.1-0.3: Small
- 0.3-0.5: Medium
- > 0.5: Large

## 🔧 Common Workflows

### Workflow 1: Survey Response Analysis
1. Load survey responses (text column)
2. Preprocess text
3. Run clustering (5-7 clusters)
4. Generate word clouds per cluster
5. Analyze sentiment
6. Compare sentiment across clusters

### Workflow 2: Topic Discovery
1. Load documents
2. Preprocess text
3. Run topic modeling (5-10 topics)
4. Review top terms per topic
5. Assign documents to dominant topics
6. Export results

### Workflow 3: Group Comparison
1. Load data with group labels
2. Preprocess and cluster
3. Run chi-square test
4. Compare distributions
5. Identify significant differences

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt --upgrade
```

### "NLTK data not found" error
```python
import nltk
nltk.download('all')
```

### Application won't start
```bash
# Check if port 8501 is in use
streamlit run app.py --server.port 8502
```

### Slow performance
- Reduce `max_features` to 500-1000
- Use fewer clusters (3-5)
- Disable lemmatization
- Process smaller batches

## 📚 Next Steps

1. **Explore Examples**: Check the `notebooks/` folder for Jupyter notebooks
2. **Read Documentation**: See `README.md` for detailed information
3. **Customize**: Modify parameters to fit your data
4. **Export Results**: Save visualizations and analysis results
5. **Iterate**: Refine analysis based on initial results

## 🎓 Learning Resources

### Understanding Your Results
- **Clustering**: Groups similar items together
- **Topic Modeling**: Discovers themes in text
- **Sentiment Analysis**: Measures emotional tone
- **Chi-Square**: Tests if distributions differ

### Best Practices
1. Always explore your data first
2. Preprocess consistently
3. Try multiple parameter settings
4. Validate results with domain knowledge
5. Document your analysis steps

## 💬 Getting Help

If you encounter issues:
1. Check this guide
2. Review error messages carefully
3. Consult the main README.md
4. Check example notebooks
5. Verify data format

## ✅ Checklist

Before starting analysis:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded
- [ ] spaCy model downloaded
- [ ] Data file prepared (CSV/Excel/TXT/JSON)
- [ ] Text column identified (for unstructured data)
- [ ] Application running (`streamlit run app.py`)

---

**Ready to analyze? Run `streamlit run app.py` and start exploring your data!** 🚀