# AI Narrative Nexus - Dynamic Text Analysis Platform

A comprehensive end-to-end text analysis platform that performs sentiment analysis, topic modeling, and insight generation on the Twitter US Airline Sentiment Dataset.

## 🌟 Features

- **Automated Text Preprocessing**: URL removal, tokenization, stopword filtering, and lemmatization
- **Sentiment Classification**: Multi-method analysis using VADER, TextBlob, and optional DistilBERT
- **Topic Modeling**: LDA and NMF algorithms for discovering hidden themes
- **Insight Generation**: Actionable insights with word clouds and automated summaries
- **Interactive Dashboard**: Modern React frontend with real-time visualizations
- **Report Export**: Comprehensive PDF reports with charts and statistics

## 📁 Project Structure

```
AI_Narrative_Nexus/
│
├── data/                          # Data directory
│   └── cleaned_dataset.csv.csv    # Input dataset (place your data here)
│
├── src/                           # Python analysis modules
│   ├── data_processing.py         # Text preprocessing pipeline
│   ├── sentiment_analysis.py      # Sentiment classification
│   ├── topic_modeling.py          # Topic extraction (LDA/NMF)
│   ├── insights_generation.py     # Insights & word clouds
│   ├── report_generator.py        # PDF report generation
│   └── main.py                    # Pipeline orchestrator
│
├── backend/                       # Flask REST API
│   ├── app.py                     # API server
│   └── requirements.txt           # Python dependencies
│
├── frontend/                      # React application
│   ├── src/
│   │   ├── pages/                 # Page components
│   │   ├── App.jsx                # Main app component
│   │   └── api.js                 # API client
│   ├── package.json               # Node dependencies
│   └── vite.config.js             # Vite configuration
│
├── reports/                       # Generated reports and visualizations
│
└── requirements.txt               # Main Python dependencies
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Backend Setup

1. **Navigate to the project root:**
   ```powershell
   cd "AI_Narrative_Nexus"
   ```

2. **Create a Python virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Python dependencies:**
   ```powershell
   pip install -r backend/requirements.txt
   ```

4. **Download NLTK data:**
   ```powershell
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

5. **Place your dataset:**
   - Copy `cleaned_dataset.csv.csv` to the `data/` directory

### Frontend Setup

1. **Navigate to frontend directory:**
   ```powershell
   cd frontend
   ```

2. **Install Node dependencies:**
   ```powershell
   npm install
   ```

## 🎯 Running the Application

### Start Backend Server

1. **Navigate to backend directory:**
   ```powershell
   cd backend
   ```

2. **Activate virtual environment (if not already active):**
   ```powershell
   ..\venv\Scripts\Activate.ps1
   ```

3. **Start Flask server:**
   ```powershell
   python app.py
   ```
   
   The API will be available at `http://localhost:5000`

### Start Frontend Development Server

1. **In a new terminal, navigate to frontend directory:**
   ```powershell
   cd frontend
   ```

2. **Start Vite development server:**
   ```powershell
   npm run dev
   ```
   
   The application will be available at `http://localhost:3000`

## 💻 Using the Application

### Method 1: Quick Start (Recommended)

1. Open `http://localhost:3000` in your browser
2. Click **"Start Full Analysis"** on the Home page
3. Wait for the complete pipeline to finish (5-10 minutes)
4. Navigate to different pages to view results
5. Download reports from the Reports page

### Method 2: Step-by-Step Analysis

1. **Home Page**: View dataset overview
2. **Preprocessing**: Clean and prepare text data
3. **Sentiment Analysis**: Classify tweet sentiments
4. **Topic Modeling**: Extract themes and topics
5. **Insights**: View actionable insights and word clouds
6. **Reports**: Generate and download PDF reports

## 📊 API Endpoints

### Dataset
- `GET /api/dataset/info` - Get dataset information
- `GET /api/health` - Health check

### Analysis
- `POST /api/preprocess` - Run text preprocessing
- `POST /api/sentiment` - Analyze sentiment
- `POST /api/topics` - Extract topics
- `POST /api/insights` - Generate insights
- `POST /api/analysis/full` - Run complete pipeline

### Reports
- `POST /api/report/generate` - Generate PDF report
- `GET /api/report/download` - Download PDF report
- `GET /api/data/download/:filename` - Download CSV files
- `GET /api/wordcloud/:sentiment` - Get word cloud image

### Status
- `GET /api/status` - Get current analysis status

## 🛠️ Alternative: Run Analysis via Command Line

You can also run the analysis pipeline directly without the web interface:

```powershell
cd src
python main.py
```

This will execute all steps and generate reports in the `reports/` directory.

## 📦 Key Dependencies

### Backend
- Flask - REST API framework
- pandas - Data manipulation
- nltk - Natural language processing
- vaderSentiment - Sentiment analysis
- gensim - Topic modeling
- plotly - Visualizations
- reportlab - PDF generation
- wordcloud - Word cloud generation

### Frontend
- React - UI framework
- Ant Design - Component library
- Recharts - Chart library
- Axios - HTTP client
- React Router - Navigation

## 🎨 Features by Page

### Home Page
- Dataset overview and statistics
- Quick start with full pipeline
- System status monitoring

### Preprocessing Page
- Text cleaning pipeline
- Before/after comparison
- Word count statistics

### Sentiment Analysis Page
- VADER + TextBlob analysis
- Optional DistilBERT classification
- Sentiment distribution charts
- Airline-wise sentiment breakdown

### Topic Modeling Page
- LDA/NMF topic extraction
- Configurable number of topics
- Topic distribution visualization
- Automatic topic labeling

### Insights Page
- Airline performance rankings
- Word clouds for each sentiment
- Top issues and positive aspects
- Automated recommendations

### Reports Page
- PDF report generation
- CSV data export
- Comprehensive analysis summary

## 📝 Configuration Options

### Sentiment Analysis
- `use_distilbert`: Enable DistilBERT model (slower but more accurate)

### Topic Modeling
- `n_topics`: Number of topics to extract (3-15)
- `method`: Algorithm to use ('lda' or 'nmf')

## 🐛 Troubleshooting

### Backend Issues

**Port already in use:**
```powershell
# Change port in backend/app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

**NLTK data missing:**
```powershell
python -c "import nltk; nltk.download('all')"
```

**Module not found:**
```powershell
pip install -r backend/requirements.txt
```

### Frontend Issues

**Port already in use:**
Edit `frontend/vite.config.js` and change the port

**API connection issues:**
Check that backend is running on `http://localhost:5000`

**Dependencies issues:**
```powershell
rm -r node_modules
npm install
```

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📧 Support

For issues and questions, please create an issue in the repository.

---

**Built with ❤️ using Flask, React, and modern NLP techniques**
