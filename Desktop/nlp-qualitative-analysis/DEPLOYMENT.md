# Deployment Guide

## Hosting on GitHub and Streamlit Cloud

This guide will help you deploy the NLP Qualitative Analysis Tool to the web so you can access it from your browser.

## Prerequisites

- GitHub account
- Git installed on your computer

## Step 1: Push to GitHub

1. **Initialize Git repository** (if not already done):
```bash
cd /Users/phildoyleibm.com/Desktop/nlp-qualitative-analysis
git init
```

2. **Add all files**:
```bash
git add .
git commit -m "Initial commit: NLP Qualitative Analysis Tool with RTA Assistant"
```

3. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name it: `nlp-qualitative-analysis`
   - Don't initialize with README (you already have one)
   - Click "Create repository"

4. **Push to GitHub**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/nlp-qualitative-analysis.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Streamlit Cloud

### Option A: Streamlit Community Cloud (Free)

1. **Go to Streamlit Cloud**:
   - Visit https://streamlit.io/cloud
   - Click "Sign up" or "Sign in" with your GitHub account

2. **Deploy your app**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/nlp-qualitative-analysis`
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Wait for deployment** (usually 2-5 minutes)
   - Streamlit will install dependencies from `requirements.txt`
   - Your app will be available at: `https://YOUR_USERNAME-nlp-qualitative-analysis.streamlit.app`

### Option B: Run Locally

If you prefer to run locally and access via browser:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Step 3: Share Your App

Once deployed to Streamlit Cloud, you can:
- Share the URL with anyone
- No authentication required (unless you add it)
- Free tier includes:
  - Unlimited public apps
  - 1 GB RAM per app
  - Community support

## Troubleshooting

### Deployment Fails

If deployment fails, check:

1. **requirements.txt** - Ensure all dependencies are listed
2. **Python version** - Streamlit Cloud uses Python 3.9+ by default
3. **File paths** - Use relative paths, not absolute paths
4. **Large files** - GitHub has 100MB file limit

### App Runs Slowly

For better performance:

1. **Cache data loading**:
```python
@st.cache_data
def load_data(file):
    return pd.read_csv(file)
```

2. **Use session state** for expensive computations
3. **Limit file upload size** in Streamlit config

### Memory Issues

If you hit memory limits:

1. **Process data in chunks**
2. **Use sampling** for large datasets
3. **Clear session state** when not needed
4. **Consider upgrading** to Streamlit Cloud Pro

## Advanced Configuration

### Custom Domain

To use a custom domain:

1. Go to app settings in Streamlit Cloud
2. Add your custom domain
3. Update DNS records as instructed

### Environment Variables

For sensitive data (API keys, etc.):

1. Go to app settings
2. Click "Secrets"
3. Add key-value pairs
4. Access in code: `st.secrets["KEY_NAME"]`

### Authentication

To add password protection:

```python
import streamlit_authenticator as stauth

# Add to app.py
authenticator = stauth.Authenticate(
    credentials,
    'cookie_name',
    'signature_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Your app code here
    pass
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

## Monitoring

### View Logs

In Streamlit Cloud:
1. Go to your app
2. Click "Manage app"
3. View logs in real-time

### Analytics

Track usage with:
- Google Analytics
- Streamlit's built-in analytics (Pro tier)
- Custom logging

## Updates

To update your deployed app:

```bash
# Make changes locally
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud will auto-deploy
```

## Cost

### Free Tier
- ✅ Unlimited public apps
- ✅ 1 GB RAM per app
- ✅ Community support
- ✅ GitHub integration

### Pro Tier ($20/month)
- ✅ Private apps
- ✅ 4 GB RAM per app
- ✅ Priority support
- ✅ Custom domains
- ✅ Analytics

## Support

- **Streamlit Docs**: https://docs.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: https://github.com/streamlit/streamlit/issues

## Security Best Practices

1. **Never commit secrets** to GitHub
2. **Use environment variables** for sensitive data
3. **Validate user inputs** to prevent injection attacks
4. **Limit file upload sizes** to prevent DoS
5. **Use HTTPS** (automatic with Streamlit Cloud)

## Next Steps

After deployment:
1. Test all features in the deployed app
2. Share the URL with your team
3. Monitor usage and performance
4. Iterate based on feedback

---

**Need help?** Check the [README.md](README.md) for feature documentation or open an issue on GitHub.