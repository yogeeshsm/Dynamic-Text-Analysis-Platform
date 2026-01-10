# Quick Start Guide - AI Narrative Nexus

## 🚀 Setup (First Time Only)

### 1. Run Setup Script
```powershell
cd "AI_Narrative_Nexus"
.\setup.ps1
```

This will:
- Create Python virtual environment
- Install all Python dependencies
- Install all Node.js dependencies
- Download required NLTK data

### 2. Add Your Dataset
Copy `cleaned_dataset.csv.csv` to the `data/` folder

## ▶️ Running the Application

### Option 1: Using Batch Scripts (Easiest)

**Terminal 1 - Backend:**
```powershell
cd backend
.\start_backend.bat
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
.\start_frontend.bat
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```powershell
cd backend
..\venv\Scripts\Activate.ps1
python app.py
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm run dev
```

### Open Application
Navigate to: **http://localhost:3000**

## 📊 Using the Platform

### Quick Analysis (Recommended)
1. Go to **Home** page
2. Click **"Start Full Analysis"**
3. Wait 5-10 minutes for completion
4. Explore results in each page

### Step-by-Step Analysis
1. **Preprocessing** → Clean text data
2. **Sentiment Analysis** → Classify sentiments
3. **Topic Modeling** → Extract topics (choose 5-7 topics)
4. **Insights** → Generate insights & word clouds
5. **Reports** → Generate & download PDF report

## 🛠️ Command Line Analysis

Run complete pipeline without web UI:
```powershell
cd src
..\venv\Scripts\Activate.ps1
python main.py
```

## 📥 Downloading Results

From the **Reports** page, you can download:
- PDF Report with all visualizations
- CSV files:
  - `clean_tweets.csv`
  - `sentiment_results.csv`
  - `topic_results.csv`
  - `topics.csv`
  - `airline_statistics.csv`

## 🔧 Troubleshooting

### Backend won't start
```powershell
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Kill process if needed
taskkill /PID <PID> /F

# Or change port in backend/app.py
```

### Frontend won't start
```powershell
# Reinstall dependencies
cd frontend
rm -r node_modules
npm install
```

### Missing NLTK data
```powershell
..\venv\Scripts\Activate.ps1
python -c "import nltk; nltk.download('all')"
```

### Import errors
```powershell
cd backend
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 📝 Configuration

### Sentiment Analysis Options
- **Use DistilBERT**: More accurate but slower (unchecked by default)

### Topic Modeling Options
- **Number of Topics**: 5-7 recommended (default: 7)
- **Method**: LDA (recommended) or NMF

## 🎯 Tips for Best Results

1. **Dataset**: Ensure your CSV has `text` and optionally `airline` columns
2. **Topics**: Start with 7 topics, adjust if results are too granular/broad
3. **Method**: Use LDA for general topics, NMF for sharper themes
4. **Reports**: Generate reports only after completing all analysis steps

## 📚 Architecture

```
User (Browser) → React Frontend (Port 3000)
                        ↓
                 Flask Backend (Port 5000)
                        ↓
            Python Analysis Modules (src/)
                        ↓
                Data & Reports (data/, reports/)
```

## 🔗 API Endpoints

- Backend API: `http://localhost:5000/api`
- Frontend: `http://localhost:3000`
- API Documentation: See `backend/app.py`

## ⚡ Performance Notes

- **Preprocessing**: ~1-2 minutes for 10k tweets
- **Sentiment Analysis**: ~2-3 minutes (VADER), ~10-15 minutes (DistilBERT)
- **Topic Modeling**: ~2-5 minutes
- **Insights**: ~1 minute
- **Report Generation**: ~30 seconds

**Total pipeline**: ~5-10 minutes for standard analysis

## 🎨 Features Summary

✅ Automated text preprocessing
✅ Multi-method sentiment analysis
✅ LDA/NMF topic modeling
✅ Interactive visualizations
✅ Word cloud generation
✅ Actionable insights
✅ PDF report export
✅ CSV data export
✅ Real-time progress tracking
✅ Airline performance rankings

---

**Need Help?** Check README.md or create an issue in the repository.
