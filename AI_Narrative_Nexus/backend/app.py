"""
Flask REST API for AI Narrative Nexus
Provides endpoints for text analysis operations
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_processing import TextPreprocessor, load_and_preprocess_data
from sentiment_analysis import SentimentAnalyzer, analyze_sentiment
from topic_modeling import TopicModeler, perform_topic_modeling
from insights_generation import InsightsGenerator, generate_insights
from report_generator import generate_pdf_report
from database import AnalysisDatabase
from text_summarization import TextSummarizer, summarize_dataset, generate_overall_summary

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DB_PATH = os.path.join(DATA_DIR, 'analysis.db')

# Initialize database
db = AnalysisDatabase(DB_PATH)
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Global state to track analysis progress
analysis_state = {
    'status': 'idle',  # idle, processing, completed, error
    'current_step': '',
    'progress': 0,
    'message': '',
    'error': None
}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/dataset/info', methods=['GET'])
def get_dataset_info():
    """Get information about the current dataset"""
    try:
        raw_data_path = os.path.join(DATA_DIR, 'uploaded_data.csv')
        
        if not os.path.exists(raw_data_path):
            return jsonify({
                'exists': False,
                'message': 'No dataset uploaded. Please upload a CSV file first.'
            }), 404
        
        # Load dataset
        df = pd.read_csv(raw_data_path)
        
        # Get statistics
        stats = {
            'exists': True,
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict('records')
        }
        
        # Get airline distribution if column exists
        if 'airline' in df.columns:
            stats['airline_distribution'] = df['airline'].value_counts().to_dict()
        
        # Get sentiment distribution if column exists
        if 'airline_sentiment' in df.columns:
            stats['sentiment_distribution'] = df['airline_sentiment'].value_counts().to_dict()
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/upload', methods=['POST'])
def upload_dataset():
    """Upload a new dataset file"""
    print("=== Upload endpoint called ===")
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'.txt', '.csv', '.docx'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not supported. Use TXT, CSV, or DOCX'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(DATA_DIR, 'uploaded_data' + file_ext)
        file.save(upload_path)
        
        # Process different file types
        if file_ext == '.txt':
            # Read text file and create DataFrame
            with open(upload_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({'text': texts})
        
        elif file_ext == '.csv':
            # Read CSV file
            df = pd.read_csv(upload_path)
            # Auto-detect text column
            text_columns = ['text', 'tweet', 'review', 'content', 'message', 'comment']
            text_col = None
            for col in df.columns:
                if col.lower() in text_columns:
                    text_col = col
                    break
            if text_col is None:
                # Use first column if no standard text column found
                text_col = df.columns[0]
            if text_col != 'text':
                df['text'] = df[text_col]
        
        elif file_ext == '.docx':
            # For DOCX, you'd need python-docx library
            # For now, return an error
            return jsonify({'error': 'DOCX support coming soon. Please use TXT or CSV format.'}), 400
        
        # Save as the active dataset
        active_dataset_path = os.path.join(DATA_DIR, 'uploaded_data.csv')
        df.to_csv(active_dataset_path, index=False)
        
        # Create new session in database
        session_id = db.create_session(
            dataset_name=filename,
            total_records=len(df),
            source_file=upload_path,
            description=f'Uploaded via web interface at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )
        
        # Save raw data to database
        db.save_raw_data(session_id, df)
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully: {filename}',
            'records': len(df),
            'columns': df.columns.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess the dataset"""
    global analysis_state
    
    try:
        analysis_state = {
            'status': 'processing',
            'current_step': 'preprocessing',
            'progress': 10,
            'message': 'Starting preprocessing...',
            'error': None
        }
        
        # Get parameters
        data = request.json or {}
        text_column = data.get('text_column', 'text')
        
        # File paths
        input_file = os.path.join(DATA_DIR, 'uploaded_data.csv')
        output_file = os.path.join(DATA_DIR, 'clean_tweets.csv')
        
        if not os.path.exists(input_file):
            analysis_state['status'] = 'error'
            analysis_state['error'] = 'Input file not found'
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Preprocess
        analysis_state['message'] = 'Preprocessing text (optimized mode)...'
        analysis_state['progress'] = 50
        
        print(f"Starting preprocessing for column: {text_column}")
        
        df_cleaned, summary = load_and_preprocess_data(
            input_file,
            output_file,
            text_column
        )
        
        analysis_state['status'] = 'completed'
        analysis_state['progress'] = 100
        analysis_state['message'] = 'Preprocessing completed'
        
        # Save preprocessed data to database
        session_id = db.get_latest_session_id()
        if session_id:
            db.save_preprocessed_data(session_id, df_cleaned)
        
        # Convert int64 and other numpy types to native Python types
        summary_clean = {}
        for key, value in summary.items():
            if isinstance(value, (np.integer, np.int64)):
                summary_clean[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                summary_clean[key] = float(value)
            elif isinstance(value, dict):
                summary_clean[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v for k, v in value.items()}
            else:
                summary_clean[key] = value
        
        # Convert preview data
        preview_data = df_cleaned[['original_text', 'cleaned_text']].head(10).to_dict('records')
        
        # Return results
        return jsonify({
            'success': True,
            'summary': summary_clean,
            'preview': preview_data,
            'total_records': int(len(df_cleaned))
        })
    
    except Exception as e:
        analysis_state['status'] = 'error'
        analysis_state['error'] = str(e)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment_api():
    """Perform sentiment analysis"""
    global analysis_state
    
    try:
        analysis_state = {
            'status': 'processing',
            'current_step': 'sentiment_analysis',
            'progress': 10,
            'message': 'Starting sentiment analysis...',
            'error': None
        }
        
        # Get parameters
        data = request.json or {}
        # Prefer SVM by default and do not use transformers unless explicitly requested
        use_svm = data.get('use_svm', True)
        use_distilbert = data.get('use_distilbert', False)
        
        # File paths
        input_file = os.path.join(DATA_DIR, 'clean_tweets.csv')
        output_file = os.path.join(DATA_DIR, 'sentiment_results.csv')
        
        if not os.path.exists(input_file):
            analysis_state['status'] = 'error'
            analysis_state['error'] = 'Preprocessed data not found. Run preprocessing first.'
            return jsonify({'error': 'Preprocessed data not found'}), 404
        
        # Analyze sentiment
        analysis_state['message'] = 'Analyzing sentiment (fast mode)...'
        analysis_state['progress'] = 50
        
        print(f"Starting sentiment analysis with SVM={use_svm}, DistilBERT={use_distilbert}")
        
        df_sentiment = analyze_sentiment(
            input_file,
            output_file,
            text_column='cleaned_text',
            use_distilbert=use_distilbert,
            use_svm=use_svm
        )
        
        analysis_state['status'] = 'completed'
        analysis_state['progress'] = 100
        analysis_state['message'] = 'Sentiment analysis completed'
        
        # Save sentiment results to database
        session_id = db.get_latest_session_id()
        if session_id:
            db.save_sentiment_results(session_id, df_sentiment)
        
        # Get statistics
        sentiment_counts = df_sentiment['sentiment_label'].value_counts().to_dict()
        sentiment_counts = {k: int(v) for k, v in sentiment_counts.items()}  # Convert to int
        avg_sentiment = float(df_sentiment['sentiment_score'].mean())
        
        # Get airline sentiment if available
        airline_sentiment = None
        if 'airline' in df_sentiment.columns:
            airline_sentiment_raw = df_sentiment.groupby('airline')['sentiment_label'].value_counts().unstack(fill_value=0).to_dict()
            # Convert nested dict values to int
            airline_sentiment = {k: {k2: int(v2) for k2, v2 in v.items()} for k, v in airline_sentiment_raw.items()}
        
        return jsonify({
            'success': True,
            'sentiment_distribution': sentiment_counts,
            'average_sentiment': avg_sentiment,
            'airline_sentiment': airline_sentiment,
            'total_records': int(len(df_sentiment)),
            'preview': df_sentiment[['cleaned_text', 'sentiment_label', 'sentiment_score']].head(10).to_dict('records')
        })
    
    except Exception as e:
        analysis_state['status'] = 'error'
        analysis_state['error'] = str(e)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/topics', methods=['POST'])
def analyze_topics_api():
    """Perform topic modeling"""
    global analysis_state
    
    try:
        analysis_state = {
            'status': 'processing',
            'current_step': 'topic_modeling',
            'progress': 10,
            'message': 'Starting topic modeling...',
            'error': None
        }
        
        # Get parameters
        data = request.json or {}
        n_topics = data.get('n_topics', 7)
        method = data.get('method', 'lda')
        
        # File paths
        input_file = os.path.join(DATA_DIR, 'sentiment_results.csv')
        output_file = os.path.join(DATA_DIR, 'topic_results.csv')
        
        if not os.path.exists(input_file):
            analysis_state['status'] = 'error'
            analysis_state['error'] = 'Sentiment data not found. Run sentiment analysis first.'
            return jsonify({'error': 'Sentiment data not found'}), 404
        
        # Perform topic modeling
        analysis_state['message'] = f'Extracting topics (fast mode: {method})...'
        analysis_state['progress'] = 50
        
        print(f"Starting topic modeling with {n_topics} topics using {method}")
        
        df_topics, topics_df = perform_topic_modeling(
            input_file,
            output_file,
            n_topics=n_topics,
            method=method,
            text_column='cleaned_text'
        )
        
        analysis_state['status'] = 'completed'
        analysis_state['progress'] = 100
        analysis_state['message'] = 'Topic modeling completed'
        
        # Save topic results to database
        session_id = db.get_latest_session_id()
        if session_id:
            db.save_topic_results(session_id, df_topics, topics_df)
        
        # Get topic distribution and convert to int
        topic_distribution = df_topics['topic_label'].value_counts().to_dict()
        topic_distribution = {k: int(v) for k, v in topic_distribution.items()}
        
        return jsonify({
            'success': True,
            'topics': topics_df.to_dict('records'),
            'topic_distribution': topic_distribution,
            'total_records': int(len(df_topics)),
            'preview': df_topics[['cleaned_text', 'topic_label', 'dominant_topic']].head(10).to_dict('records')
        })
    
    except Exception as e:
        analysis_state['status'] = 'error'
        analysis_state['error'] = str(e)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/insights', methods=['POST'])
def generate_insights_api():
    """Generate insights from analysis"""
    global analysis_state
    
    try:
        analysis_state = {
            'status': 'processing',
            'current_step': 'insights_generation',
            'progress': 10,
            'message': 'Generating insights...',
            'error': None
        }
        
        # File paths
        input_file = os.path.join(DATA_DIR, 'topic_results.csv')
        
        if not os.path.exists(input_file):
            analysis_state['status'] = 'error'
            analysis_state['error'] = 'Topic data not found. Run topic modeling first.'
            return jsonify({'error': 'Topic data not found'}), 404
        
        # Generate insights
        analysis_state['message'] = 'Analyzing data...'
        analysis_state['progress'] = 50
        
        insights = generate_insights(input_file, REPORTS_DIR, airline_column='airline')
        
        analysis_state['status'] = 'completed'
        analysis_state['progress'] = 100
        analysis_state['message'] = 'Insights generated'
        
        # Save insights to database
        session_id = db.get_latest_session_id()
        if session_id:
            db.save_insights(session_id, insights)
        
        # Prepare response with comprehensive insights
        response_data = {
            'success': True,
            'summary_text': insights.get('summary_text', ''),
        }
        
        # Add overall statistics
        if 'overall_statistics' in insights:
            stats = insights['overall_statistics']
            response_data['overall_statistics'] = {
                'total_records': stats['total_records'],
                'sentiment_counts': stats['sentiment_counts'],
                'sentiment_percentages': stats['sentiment_percentages'],
                'avg_text_length': stats['avg_text_length'],
                'avg_word_count': stats['avg_word_count'],
                'total_unique_keywords': stats['total_unique_keywords'],
                'top_keywords': [{'keyword': k, 'count': int(c)} for k, c in stats['top_keywords'][:30]],
            }
            
            if 'total_airlines' in stats:
                response_data['overall_statistics']['total_airlines'] = stats['total_airlines']
                response_data['overall_statistics']['most_mentioned_airline'] = stats['most_mentioned_airline']
                response_data['overall_statistics']['airline_mentions'] = stats['airline_mentions']
        
        # Add keywords by sentiment
        if 'keywords_by_sentiment' in insights:
            response_data['keywords_by_sentiment'] = {
                sentiment: [{'keyword': k, 'count': int(c)} for k, c in keywords]
                for sentiment, keywords in insights['keywords_by_sentiment'].items()
            }
        
        # Add airline sentiment if available
        if 'airline_sentiment' in insights:
            airline_df = insights['airline_sentiment'].reset_index()
            response_data['airline_rankings'] = [
                {
                    'airline': row[analysis_state.get('airline_column', 'airline')],
                    'negative_count': int(row['negative_count']),
                    'neutral_count': int(row['neutral_count']),
                    'positive_count': int(row['positive_count']),
                    'negative_pct': float(row['negative_pct']),
                    'neutral_pct': float(row['neutral_pct']),
                    'positive_pct': float(row['positive_pct']),
                    'avg_sentiment': float(row['avg_sentiment']),
                    'total_mentions': int(row['total_mentions']),
                    'rank': int(row['rank'])
                }
                for _, row in airline_df.iterrows()
            ]
        
        # Add top issues if available
        if 'top_issues' in insights:
            response_data['top_issues'] = insights['top_issues']
        
        # Add top positives if available
        if 'top_positives' in insights:
            response_data['top_positives'] = insights['top_positives']
        
        return jsonify(response_data)
    
    except Exception as e:
        analysis_state['status'] = 'error'
        analysis_state['error'] = str(e)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/summary/extractive', methods=['POST'])
def generate_extractive_summary():
    """Generate extractive summaries"""
    global analysis_state
    
    try:
        data = request.get_json()
        num_sentences = data.get('num_sentences', 3)
        
        analysis_state = {
            'status': 'processing',
            'current_step': 'extractive_summary',
            'progress': 10,
            'message': 'Generating extractive summaries...',
            'error': None
        }
        
        # File paths
        input_file = os.path.join(DATA_DIR, 'sentiment_results.csv')
        
        if not os.path.exists(input_file):
            analysis_state['status'] = 'error'
            analysis_state['error'] = 'Sentiment data not found. Run sentiment analysis first.'
            return jsonify({'error': 'Sentiment data not found'}), 404
        
        # Load data
        df = pd.read_csv(input_file)
        
        # Determine text column
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
        sentiment_column = 'sentiment' if 'sentiment' in df.columns else None
        
        analysis_state['message'] = 'Generating individual summaries...'
        analysis_state['progress'] = 30
        
        # Generate summaries for each row
        df = summarize_dataset(df, text_column=text_column, summary_type='extractive', 
                              num_sentences=num_sentences)
        
        analysis_state['message'] = 'Generating overall summaries by sentiment...'
        analysis_state['progress'] = 60
        
        # Generate overall summaries by sentiment
        overall_summaries = generate_overall_summary(df, text_column=text_column,
                                                     sentiment_column=sentiment_column,
                                                     summary_type='extractive')
        
        # Save results
        output_file = os.path.join(DATA_DIR, 'extractive_summaries.csv')
        df.to_csv(output_file, index=False)
        
        analysis_state['status'] = 'completed'
        analysis_state['progress'] = 100
        analysis_state['message'] = 'Extractive summaries generated'
        
        return jsonify({
            'success': True,
            'overall_summaries': overall_summaries,
            'total_summaries': len(df),
            'sample_summaries': df[['extractive_summary']].head(10).to_dict('records')
        })
    
    except Exception as e:
        analysis_state['status'] = 'error'
        analysis_state['error'] = str(e)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/summary/abstractive', methods=['POST'])
def generate_abstractive_summary():
    """Generate abstractive summaries"""
    global analysis_state
    
    try:
        data = request.get_json()
        max_words = data.get('max_words', 50)
        
        analysis_state = {
            'status': 'processing',
            'current_step': 'abstractive_summary',
            'progress': 10,
            'message': 'Generating abstractive summaries...',
            'error': None
        }
        
        # File paths
        input_file = os.path.join(DATA_DIR, 'sentiment_results.csv')
        
        if not os.path.exists(input_file):
            analysis_state['status'] = 'error'
            analysis_state['error'] = 'Sentiment data not found. Run sentiment analysis first.'
            return jsonify({'error': 'Sentiment data not found'}), 404
        
        # Load data
        df = pd.read_csv(input_file)
        
        # Determine text column
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
        sentiment_column = 'sentiment' if 'sentiment' in df.columns else None
        
        analysis_state['message'] = 'Generating individual summaries...'
        analysis_state['progress'] = 30
        
        # Generate summaries for each row
        df = summarize_dataset(df, text_column=text_column, summary_type='abstractive', 
                              max_words=max_words)
        
        analysis_state['message'] = 'Generating overall summaries by sentiment...'
        analysis_state['progress'] = 60
        
        # Generate overall summaries by sentiment
        overall_summaries = generate_overall_summary(df, text_column=text_column,
                                                     sentiment_column=sentiment_column,
                                                     summary_type='abstractive')
        
        # Save results
        output_file = os.path.join(DATA_DIR, 'abstractive_summaries.csv')
        df.to_csv(output_file, index=False)
        
        analysis_state['status'] = 'completed'
        analysis_state['progress'] = 100
        analysis_state['message'] = 'Abstractive summaries generated'
        
        return jsonify({
            'success': True,
            'overall_summaries': overall_summaries,
            'total_summaries': len(df),
            'sample_summaries': df[['abstractive_summary']].head(10).to_dict('records')
        })
    
    except Exception as e:
        analysis_state['status'] = 'error'
        analysis_state['error'] = str(e)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/report/generate', methods=['POST'])
def generate_report_api():
    """Generate PDF report"""
    try:
        # File paths
        data_file = os.path.join(DATA_DIR, 'topic_results.csv')
        topics_file = os.path.join(DATA_DIR, 'topics.csv')
        insights_file = os.path.join(REPORTS_DIR, 'insights_summary.txt')
        extractive_summaries_file = os.path.join(DATA_DIR, 'extractive_summaries.csv')
        abstractive_summaries_file = os.path.join(DATA_DIR, 'abstractive_summaries.csv')
        output_path = os.path.join(REPORTS_DIR, 'airline_sentiment_report.pdf')
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'Analysis data not found. Run complete analysis first.'}), 404
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Load topics if available
        topics_df = None
        if os.path.exists(topics_file):
            topics_df = pd.read_csv(topics_file)
        
        # Load insights text if available
        insights_text = None
        if os.path.exists(insights_file):
            with open(insights_file, 'r', encoding='utf-8') as f:
                insights_text = f.read()
        
        # Load summaries if available
        summaries_dict = {}
        
        # Load extractive summaries
        if os.path.exists(extractive_summaries_file):
            extractive_df = pd.read_csv(extractive_summaries_file)
            text_column = 'cleaned_text' if 'cleaned_text' in extractive_df.columns else 'text'
            sentiment_column = 'sentiment' if 'sentiment' in extractive_df.columns else None
            
            if sentiment_column:
                extractive_summaries = generate_overall_summary(
                    extractive_df, 
                    text_column=text_column,
                    sentiment_column=sentiment_column,
                    summary_type='extractive'
                )
                summaries_dict['extractive'] = extractive_summaries
        
        # Load abstractive summaries
        if os.path.exists(abstractive_summaries_file):
            abstractive_df = pd.read_csv(abstractive_summaries_file)
            text_column = 'cleaned_text' if 'cleaned_text' in abstractive_df.columns else 'text'
            sentiment_column = 'sentiment' if 'sentiment' in abstractive_df.columns else None
            
            if sentiment_column:
                abstractive_summaries = generate_overall_summary(
                    abstractive_df,
                    text_column=text_column,
                    sentiment_column=sentiment_column,
                    summary_type='abstractive'
                )
                summaries_dict['abstractive'] = abstractive_summaries
        
        # Generate report
        success = generate_pdf_report(
            df,
            topics_df=topics_df,
            insights_text=insights_text,
            output_path=output_path,
            wordcloud_dir=REPORTS_DIR,
            summaries_dict=summaries_dict if summaries_dict else None
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Report generated successfully',
                'file_path': output_path
            })
        else:
            return jsonify({'error': 'Report generation failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/report/download', methods=['GET'])
def download_report():
    """Download the generated PDF report"""
    try:
        report_path = os.path.join(REPORTS_DIR, 'airline_sentiment_report.pdf')
        
        if not os.path.exists(report_path):
            return jsonify({'error': 'Report not found. Generate report first.'}), 404
        
        return send_file(
            report_path,
            as_attachment=True,
            download_name='airline_sentiment_report.pdf',
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/dashboard', methods=['POST'])
def generate_dashboard_api():
    """Generate interactive HTML dashboard with enhanced visualizations"""
    try:
        # Import from src directory
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
        from report_generator import generate_interactive_dashboard
        
        # File paths
        data_file = os.path.join(DATA_DIR, 'topic_results.csv')
        topics_file = os.path.join(DATA_DIR, 'topics.csv')
        insights_file = os.path.join(REPORTS_DIR, 'insights_summary.txt')
        output_path = os.path.join(REPORTS_DIR, 'interactive_dashboard.html')
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'Analysis data not found. Run complete analysis first.'}), 404
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Add computed columns if missing
        if 'word_count' not in df.columns and 'cleaned_text' in df.columns:
            df['word_count'] = df['cleaned_text'].fillna('').str.split().str.len()
        
        if 'text_length' not in df.columns and 'cleaned_text' in df.columns:
            df['text_length'] = df['cleaned_text'].fillna('').str.len()
        
        # Load topics if available
        topics_df = None
        if os.path.exists(topics_file):
            topics_df = pd.read_csv(topics_file)
        
        # Load insights text if available
        insights_text = None
        if os.path.exists(insights_file):
            with open(insights_file, 'r', encoding='utf-8') as f:
                insights_text = f.read()
        
        # Generate dashboard
        dashboard_path = generate_interactive_dashboard(
            df,
            topics_df=topics_df,
            insights_text=insights_text,
            output_path=output_path
        )
        
        return jsonify({
            'success': True,
            'message': 'Interactive dashboard generated successfully',
            'file_path': dashboard_path,
            'dashboard_url': f'/reports/interactive_dashboard.html'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/model/save', methods=['POST'])
def save_model_api():
    """Save trained SVM model to disk"""
    try:
        from sentiment_analysis import SentimentAnalyzer
        
        # Load existing data to retrain if needed
        data_file = os.path.join(DATA_DIR, 'sentiment_results.csv')
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'No trained model available. Run sentiment analysis first.'}), 404
        
        df = pd.read_csv(data_file)
        
        # Initialize analyzer and train model
        analyzer = SentimentAnalyzer(use_svm=True)
        
        if 'airline_sentiment' in df.columns and 'cleaned_text' in df.columns:
            texts = df['cleaned_text'].fillna('').tolist()
            labels = df['airline_sentiment'].tolist()
            accuracy = analyzer.train_svm_classifier(texts, labels)
            
            # Save model
            model_dir = os.path.join(BASE_DIR, 'models')
            saved_paths = analyzer.save_model(model_dir)
            
            return jsonify({
                'success': True,
                'message': 'Model saved successfully',
                'accuracy': float(accuracy),
                'model_files': saved_paths
            })
        else:
            return jsonify({'error': 'Required columns not found in data'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model_api():
    """Load trained SVM model from disk"""
    try:
        from sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(use_svm=True)
        model_dir = os.path.join(BASE_DIR, 'models')
        
        success = analyzer.load_model(model_dir)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model loaded successfully',
                'model_dir': model_dir
            })
        else:
            return jsonify({'error': 'Failed to load model'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info_api():
    """Get information about saved models"""
    try:
        model_dir = os.path.join(BASE_DIR, 'models')
        
        if not os.path.exists(model_dir):
            return jsonify({
                'exists': False,
                'message': 'No saved models found'
            })
        
        model_files = {
            'svm_model': os.path.join(model_dir, 'svm_sentiment_model.pkl'),
            'vectorizer': os.path.join(model_dir, 'tfidf_vectorizer.pkl'),
            'label_encoder': os.path.join(model_dir, 'label_encoder.pkl')
        }
        
        file_info = {}
        all_exist = True
        
        for name, path in model_files.items():
            if os.path.exists(path):
                stat = os.stat(path)
                file_info[name] = {
                    'exists': True,
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                }
            else:
                file_info[name] = {'exists': False}
                all_exist = False
        
        return jsonify({
            'exists': all_exist,
            'model_dir': model_dir,
            'files': file_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/download/<filename>', methods=['GET'])
def download_data(filename):
    """Download CSV data files"""
    try:
        allowed_files = [
            'clean_tweets.csv',
            'sentiment_results.csv',
            'topic_results.csv',
            'topics.csv',
            'airline_statistics.csv'
        ]
        
        if filename not in allowed_files:
            return jsonify({'error': 'File not allowed'}), 403
        
        file_path = os.path.join(DATA_DIR if filename != 'airline_statistics.csv' else REPORTS_DIR, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/wordcloud/<sentiment>', methods=['GET'])
def get_wordcloud(sentiment):
    """Get word cloud image for a sentiment"""
    try:
        if sentiment not in ['positive', 'negative', 'neutral']:
            return jsonify({'error': 'Invalid sentiment'}), 400
        
        image_path = os.path.join(REPORTS_DIR, f'wordcloud_{sentiment}.png')
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Word cloud not found. Generate insights first.'}), 404
        
        return send_file(image_path, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current analysis status"""
    return jsonify(analysis_state)


@app.route('/api/analysis/full', methods=['POST'])
def run_full_analysis():
    """Run complete analysis pipeline"""
    global analysis_state
    
    try:
        analysis_state = {
            'status': 'processing',
            'current_step': 'full_pipeline',
            'progress': 0,
            'message': 'Starting full analysis...',
            'error': None
        }
        
        # Get parameters
        data = request.json or {}
        use_svm = data.get('use_svm', True)
        use_distilbert = data.get('use_distilbert', False)
        n_topics = data.get('n_topics', 7)
        topic_method = data.get('topic_method', 'lda')
        
        # Step 1: Preprocessing
        analysis_state['current_step'] = 'preprocessing'
        analysis_state['progress'] = 10
        analysis_state['message'] = 'Preprocessing data...'
        
        input_file = os.path.join(DATA_DIR, 'uploaded_data.csv')
        clean_file = os.path.join(DATA_DIR, 'clean_tweets.csv')
        
        df_cleaned, _ = load_and_preprocess_data(input_file, clean_file, 'text')
        
        # Step 2: Sentiment Analysis
        analysis_state['current_step'] = 'sentiment_analysis'
        analysis_state['progress'] = 30
        analysis_state['message'] = 'Analyzing sentiment...'
        
        sentiment_file = os.path.join(DATA_DIR, 'sentiment_results.csv')
        df_sentiment = analyze_sentiment(clean_file, sentiment_file, use_distilbert=use_distilbert, use_svm=use_svm)
        
        # Step 3: Topic Modeling
        analysis_state['current_step'] = 'topic_modeling'
        analysis_state['progress'] = 60
        analysis_state['message'] = 'Extracting topics...'
        
        topic_file = os.path.join(DATA_DIR, 'topic_results.csv')
        df_topics, topics_df = perform_topic_modeling(sentiment_file, topic_file, n_topics=n_topics, method=topic_method)
        
        # Step 4: Insights Generation
        analysis_state['current_step'] = 'insights_generation'
        analysis_state['progress'] = 80
        analysis_state['message'] = 'Generating insights...'
        
        insights = generate_insights(topic_file, REPORTS_DIR)
        
        # Step 5: Report Generation
        analysis_state['current_step'] = 'report_generation'
        analysis_state['progress'] = 90
        analysis_state['message'] = 'Creating report...'
        
        report_path = os.path.join(REPORTS_DIR, 'airline_sentiment_report.pdf')
        topics_file = os.path.join(DATA_DIR, 'topics.csv')
        insights_file = os.path.join(REPORTS_DIR, 'insights_summary.txt')
        
        topics_df_loaded = pd.read_csv(topics_file) if os.path.exists(topics_file) else None
        insights_text = open(insights_file, 'r', encoding='utf-8').read() if os.path.exists(insights_file) else None
        
        generate_pdf_report(df_topics, topics_df_loaded, insights_text, report_path, REPORTS_DIR)
        
        analysis_state['status'] = 'completed'
        analysis_state['progress'] = 100
        analysis_state['message'] = 'Analysis completed successfully'
        
        return jsonify({
            'success': True,
            'message': 'Full analysis completed',
            'total_records': len(df_topics)
        })
    
    except Exception as e:
        analysis_state['status'] = 'error'
        analysis_state['error'] = str(e)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/sessions/list', methods=['GET'])
def list_sessions():
    """List all analysis sessions from database"""
    try:
        sessions_df = db.list_sessions()
        sessions = sessions_df.to_dict('records')
        
        # Convert datetime to string
        for session in sessions:
            if 'created_at' in session:
                session['created_at'] = str(session['created_at'])
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'total_sessions': len(sessions)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/<int:session_id>', methods=['GET'])
def get_session(session_id):
    """Get data for a specific session"""
    try:
        session_data = db.get_session_data(session_id)
        
        # Convert datetime fields to string
        if 'created_at' in session_data['session_info']:
            session_data['session_info']['created_at'] = str(session_data['session_info']['created_at'])
        
        # Convert DataFrame to records
        session_data['data'] = session_data['data'].to_dict('records')
        
        return jsonify({
            'success': True,
            'session': session_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and all its data"""
    try:
        db.delete_session(session_id)
        
        return jsonify({
            'success': True,
            'message': f'Session {session_id} deleted successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting AI Narrative Nexus API Server...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Database: {DB_PATH}")
    print("\nRegistered API Endpoints:")
    print("  POST   /api/dataset/upload")
    print("  GET    /api/dataset/info")
    print("  POST   /api/preprocess")
    print("  POST   /api/sentiment")
    print("  POST   /api/topics")
    print("  POST   /api/insights")
    print("  POST   /api/summary/extractive")
    print("  POST   /api/summary/abstractive")
    print("  POST   /api/report/generate")
    print("  GET    /api/report/download")
    print("  GET    /api/health")
    print("  GET    /api/status")
    print("  POST   /api/analysis/full")
    print("  GET    /api/sessions/list")
    print("  GET    /api/sessions/<id>")
    print("  DELETE /api/sessions/<id>")
    print("\nServer ready at http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
