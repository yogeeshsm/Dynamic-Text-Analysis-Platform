# Dynamic Text Analysis Platform

A comprehensive web-based platform for advanced text analytics, featuring sentiment analysis, topic modeling, text summarization, and interactive visualizations.

## 🚀 Features

- **Sentiment Analysis**: Analyze text sentiment using SVM-based machine learning models
- **Topic Modeling**: Discover hidden topics and themes in text data using LDA
- **Text Summarization**: Generate both extractive and abstractive summaries
- **Interactive Visualizations**: Beautiful charts and graphs using Plotly
- **Data Processing**: Clean and preprocess text data with advanced NLP techniques
- **Report Generation**: Export comprehensive analysis reports
- **Database Integration**: SQLite database for efficient data storage
- **Modern UI**: React-based frontend with responsive design

## 📋 Project Structure

```
dynamic-text-analysis-platform/
├── AI_Narrative_Nexus/          # Main application
│   ├── backend/                 # Flask backend API
│   ├── frontend/                # React frontend
│   ├── src/                     # Core Python modules
│   ├── data/                    # Data storage
│   ├── models/                  # ML models
│   └── reports/                 # Generated reports
├── cleaned_dataset.csv.csv      # Sample dataset
├── Sentiment_analysis.ipynb     # Jupyter notebook for sentiment analysis
├── Topic_modellingfinal.ipynb   # Jupyter notebook for topic modeling
└── Visualization_and_Reporting.ipynb  # Jupyter notebook for visualizations
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

```bash
cd AI_Narrative_Nexus/backend
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd AI_Narrative_Nexus/frontend
npm install
```

## 🚀 Quick Start

### Option 1: Use the Start Script (Windows)

```bash
cd AI_Narrative_Nexus
start_servers.bat
```

### Option 2: Manual Start

**Start Backend:**
```bash
cd AI_Narrative_Nexus/backend
python app.py
```

**Start Frontend:**
```bash
cd AI_Narrative_Nexus/frontend
npm run dev
```

The application will be available at:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:5000`

## 📊 Usage

1. **Upload Data**: Upload your text data (CSV, TXT, or DOCX)
2. **Preprocess**: Clean and prepare your data
3. **Analyze Sentiment**: Run sentiment analysis on your text
4. **Topic Modeling**: Discover topics and themes
5. **Generate Summaries**: Create extractive or abstractive summaries
6. **View Insights**: Explore interactive visualizations
7. **Generate Reports**: Export comprehensive analysis reports

## 🔧 Technologies Used

### Backend
- **Flask**: Web framework
- **Scikit-learn**: Machine learning (SVM, TF-IDF)
- **NLTK**: Natural language processing
- **Gensim**: Topic modeling (LDA)
- **Transformers**: Abstractive summarization
- **Plotly**: Interactive visualizations
- **SQLite**: Database
- **Pandas**: Data manipulation

### Frontend
- **React**: UI framework
- **Vite**: Build tool
- **Axios**: HTTP client
- **CSS3**: Styling

## 📝 Features in Detail

### Sentiment Analysis
- SVM-based classification
- TF-IDF vectorization
- Sentiment scoring and distribution
- Airline-specific sentiment analysis

### Topic Modeling
- LDA (Latent Dirichlet Allocation)
- Customizable number of topics
- Topic keyword extraction
- Interactive topic distribution charts

### Text Summarization
- **Extractive**: Key sentence extraction
- **Abstractive**: AI-generated summaries using transformers
- Customizable summary length

### Visualizations
- Sentiment distribution charts
- Topic distribution plots
- Word clouds
- Airline comparison heatmaps
- Interactive dashboards

## 📂 Key Files

- `backend/app.py`: Flask API server
- `src/sentiment_analysis.py`: Sentiment analysis module
- `src/topic_modeling.py`: Topic modeling module
- `src/text_summarization.py`: Text summarization module
- `src/report_generator.py`: Report generation module
- `frontend/src/App.jsx`: Main React application

## 🔐 Database

The platform uses SQLite for data storage. Database schema includes:
- Uploaded data
- Sentiment results
- Topic results
- Summaries

## 📈 Performance Optimizations

- Efficient batch processing
- Caching mechanisms
- Optimized database queries
- Lazy loading for large datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**S M Yogesh**
- GitHub: [@yogeeshsm](https://github.com/yogeeshsm)

## 🙏 Acknowledgments

- Scikit-learn for machine learning tools
- Hugging Face for transformer models
- Plotly for visualization library
- React community for frontend tools

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ by S M Yogesh