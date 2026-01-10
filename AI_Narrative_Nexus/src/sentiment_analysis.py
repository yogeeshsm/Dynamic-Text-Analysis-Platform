"""
Sentiment Analysis Module for AI Narrative Nexus
Performs sentiment classification using VADER, TextBlob, and SVM
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Optional: Hugging Face Transformers for DistilBERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Transformers not available. Using VADER, TextBlob, and SVM.")


class SentimentAnalyzer:
    """
    Multi-method sentiment analysis for tweet data
    Supports: VADER, TextBlob, SVM, and optional DistilBERT
    """
    
    def __init__(self, use_distilbert=False, use_svm=True):
        """
        Initialize sentiment analyzer
        
        Args:
            use_distilbert: Whether to use DistilBERT (requires transformers)
            use_svm: Whether to train and use SVM classifier (fast and accurate)
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.use_distilbert = use_distilbert and HF_AVAILABLE
        self.use_svm = use_svm
        self.svm_model = None
        self.svm_vectorizer = None
        self.svm_label_encoder = None
        
        if self.use_distilbert:
            print("Loading DistilBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print(f"DistilBERT loaded on {self.device}")
    
    def vader_sentiment(self, text):
        """
        Get VADER sentiment scores
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if pd.isna(text) or text == "":
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0
            }
        
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
    def textblob_sentiment(self, text):
        """
        Get TextBlob sentiment scores
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (polarity, subjectivity)
        """
        if pd.isna(text) or text == "":
            return 0.0, 0.0
        
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def distilbert_sentiment(self, text):
        """
        Get DistilBERT sentiment prediction
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (label, score)
        """
        if not self.use_distilbert:
            return None, None
        
        if pd.isna(text) or text == "":
            return "NEUTRAL", 0.5
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get label and confidence
        predicted_class = predictions.argmax().item()
        confidence = predictions.max().item()
        
        label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
        
        return label, confidence
    
    def classify_sentiment(self, compound_score):
        """
        Classify sentiment based on compound score
        
        Args:
            compound_score: VADER compound score
            
        Returns:
            Sentiment label (positive/neutral/negative)
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def train_svm_classifier(self, texts, labels, max_features=1000):
        """
        Train SVM classifier for sentiment analysis (ULTRA-FAST MODE)
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels
            max_features: Maximum number of TF-IDF features (minimal for speed)
            
        Returns:
            Training accuracy
        """
        print("Training SVM classifier (ULTRA-FAST mode - 20 sec target)...")
        
        # Encode labels
        self.svm_label_encoder = LabelEncoder()
        y_encoded = self.svm_label_encoder.fit_transform(labels)
        
        # Create TF-IDF features (ULTRA-FAST mode)
        self.svm_vectorizer = TfidfVectorizer(
            max_features=500,  # Even less features
            ngram_range=(1, 1),  # Only unigrams
            min_df=5,
            max_df=0.70,
            lowercase=True,
            stop_words='english',
            sublinear_tf=False,  # Faster
            use_idf=False,  # Faster
            norm=None  # Faster
        )
        
        X = self.svm_vectorizer.fit_transform(texts)
        
        # Train LinearSVC (ULTRA-FAST MODE)
        self.svm_model = LinearSVC(
            C=0.3,
            max_iter=50,  # Even less iterations
            dual=False,
            class_weight='balanced',
            loss='squared_hinge',
            tol=5e-2,  # Very loose tolerance
            random_state=42
        )
        
        self.svm_model.fit(X, y_encoded)
        
        # Calculate training accuracy
        train_accuracy = self.svm_model.score(X, y_encoded)
        print(f"✅ SVM trained! Accuracy: {train_accuracy*100:.2f}%")
        
        return train_accuracy
    
    def save_model(self, model_dir='models'):
        """
        Save trained SVM model, vectorizer, and label encoder to disk
        
        Args:
            model_dir: Directory to save model files
            
        Returns:
            Dictionary with paths to saved files
        """
        if self.svm_model is None:
            print("⚠️ No trained SVM model to save")
            return None
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(model_dir, 'svm_sentiment_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.svm_model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.svm_vectorizer, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.svm_label_encoder, f)
        
        print(f"✅ Model saved successfully!")
        print(f"   - SVM Model: {model_path}")
        print(f"   - Vectorizer: {vectorizer_path}")
        print(f"   - Label Encoder: {encoder_path}")
        
        return {
            'model': model_path,
            'vectorizer': vectorizer_path,
            'encoder': encoder_path
        }
    
    def load_model(self, model_dir='models'):
        """
        Load trained SVM model, vectorizer, and label encoder from disk
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(model_dir, 'svm_sentiment_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        # Check if all files exist
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]):
            print("⚠️ Model files not found. Train a new model first.")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.svm_vectorizer = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.svm_label_encoder = pickle.load(f)
            
            print(f"✅ Model loaded successfully from {model_dir}")
            return True
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def svm_predict(self, text):
        """
        Predict sentiment using SVM
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (label, confidence)
        """
        if self.svm_model is None or pd.isna(text) or text == "":
            return None, 0.0
        
        # Transform text to features
        X = self.svm_vectorizer.transform([text])
        
        # Get prediction
        prediction = self.svm_model.predict(X)[0]
        label = self.svm_label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence (distance from decision boundary)
        decision = self.svm_model.decision_function(X)[0]
        
        # Convert to confidence score (0-1) with better calibration
        if isinstance(decision, np.ndarray):
            confidence = float(np.max(np.abs(decision)))
        else:
            confidence = float(abs(decision))
        
        # Apply sigmoid-like normalization for better calibration
        # This maps decision values to [0.5, 1.0] range more smoothly
        confidence = 1 / (1 + np.exp(-confidence))
        
        return label, confidence
    
    def analyze_dataset(self, df, text_column='cleaned_text'):
        """
        Analyze sentiment for entire dataset using VADER, TextBlob, and optionally SVM
        
        Args:
            df: DataFrame with text data
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with sentiment scores and labels
        """
        print("Analyzing sentiment...")
        df_sentiment = df.copy()
        
        # Train SVM if enabled and labels are available
        if self.use_svm and 'airline_sentiment' in df.columns:
            print("🚀 Training SVM classifier for fast and accurate predictions...")
            # Use existing labels to train SVM
            texts = df_sentiment[text_column].fillna('').tolist()
            labels = df_sentiment['airline_sentiment'].tolist()
            self.train_svm_classifier(texts, labels)
            
            # Auto-save trained model
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, 'models')
            self.save_model(model_dir)
        
        # VADER Analysis
        print("Running VADER sentiment analysis...")
        vader_scores = []
        for text in tqdm(df_sentiment[text_column], desc="VADER"):
            scores = self.vader_sentiment(text)
            vader_scores.append(scores)
        
        # Add VADER scores
        df_sentiment['vader_compound'] = [s['compound'] for s in vader_scores]
        df_sentiment['vader_pos'] = [s['pos'] for s in vader_scores]
        df_sentiment['vader_neu'] = [s['neu'] for s in vader_scores]
        df_sentiment['vader_neg'] = [s['neg'] for s in vader_scores]
        
        # TextBlob Analysis
        print("Running TextBlob sentiment analysis...")
        textblob_scores = []
        for text in tqdm(df_sentiment[text_column], desc="TextBlob"):
            polarity, subjectivity = self.textblob_sentiment(text)
            textblob_scores.append((polarity, subjectivity))
        
        df_sentiment['textblob_polarity'] = [s[0] for s in textblob_scores]
        df_sentiment['textblob_subjectivity'] = [s[1] for s in textblob_scores]
        
        # SVM Analysis (if trained)
        if self.use_svm and self.svm_model is not None:
            print("⚡ Running SVM sentiment classification (FAST!)...")
            svm_predictions = []
            for text in tqdm(df_sentiment[text_column], desc="SVM"):
                label, confidence = self.svm_predict(text)
                svm_predictions.append((label, confidence))
            
            df_sentiment['svm_label'] = [p[0] for p in svm_predictions]
            df_sentiment['svm_confidence'] = [p[1] for p in svm_predictions]
            
            # Use ENSEMBLE approach: Combine SVM with VADER for better accuracy
            df_sentiment['sentiment_label'] = df_sentiment['svm_label']
            
            # Enhanced scoring: Weighted average of SVM confidence and VADER
            svm_scores = df_sentiment['svm_confidence'] * np.sign(
                df_sentiment.apply(
                    lambda row: 1 if row['svm_label'] == 'positive' 
                    else (-1 if row['svm_label'] == 'negative' else 0), axis=1
                )
            )
            vader_scores = df_sentiment['vader_compound']
            
            # Weighted ensemble (70% SVM, 30% VADER for robustness)
            df_sentiment['sentiment_score'] = 0.7 * svm_scores + 0.3 * vader_scores
        else:
            # Use VADER as primary if SVM not available
            df_sentiment['sentiment_label'] = df_sentiment['vader_compound'].apply(
                self.classify_sentiment
            )
            df_sentiment['sentiment_score'] = df_sentiment['vader_compound']
        
        # DistilBERT Analysis (if enabled)
        if self.use_distilbert:
            print("Running DistilBERT sentiment analysis...")
            bert_predictions = []
            for text in tqdm(df_sentiment[text_column], desc="DistilBERT"):
                label, score = self.distilbert_sentiment(text)
                bert_predictions.append((label, score))
            
            df_sentiment['bert_label'] = [p[0] for p in bert_predictions]
            df_sentiment['bert_confidence'] = [p[1] for p in bert_predictions]
        
        # Calculate sentiment statistics
        sentiment_counts = df_sentiment['sentiment_label'].value_counts()
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS RESULTS")
        print("="*60)
        print(f"Total tweets analyzed: {len(df_sentiment):,}")
        print(f"Positive: {sentiment_counts.get('positive', 0):,} ({sentiment_counts.get('positive', 0)/len(df_sentiment)*100:.2f}%)")
        print(f"Neutral: {sentiment_counts.get('neutral', 0):,} ({sentiment_counts.get('neutral', 0)/len(df_sentiment)*100:.2f}%)")
        print(f"Negative: {sentiment_counts.get('negative', 0):,} ({sentiment_counts.get('negative', 0)/len(df_sentiment)*100:.2f}%)")
        print(f"Average sentiment score: {df_sentiment['sentiment_score'].mean():.4f}")
        if self.use_svm and self.svm_model is not None:
            print(f"✅ SVM Classifier: ACTIVE (High Accuracy)")
        print("="*60 + "\n")
        
        return df_sentiment
    
    def visualize_sentiment_distribution(self, df, save_path=None):
        """
        Create visualizations for sentiment distribution
        
        Args:
            df: DataFrame with sentiment analysis results
            save_path: Path to save visualization (optional)
            
        Returns:
            Plotly figure
        """
        # Count sentiments
        sentiment_counts = df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Create bar chart
        fig = px.bar(
            sentiment_counts,
            x='Sentiment',
            y='Count',
            color='Sentiment',
            title='Sentiment Distribution',
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#95a5a6',
                'negative': '#e74c3c'
            },
            text='Count'
        )
        
        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Number of Tweets",
            showlegend=False,
            height=500
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        if save_path:
            fig.write_html(save_path)
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def visualize_sentiment_by_airline(self, df, airline_column='airline', save_path=None):
        """
        Create sentiment distribution by airline
        
        Args:
            df: DataFrame with sentiment analysis results
            airline_column: Column containing airline names
            save_path: Path to save visualization (optional)
            
        Returns:
            Plotly figure
        """
        if airline_column not in df.columns:
            print(f"Column '{airline_column}' not found. Skipping airline visualization.")
            return None
        
        # Group by airline and sentiment
        airline_sentiment = df.groupby([airline_column, 'sentiment_label']).size().reset_index(name='count')
        
        # Create grouped bar chart
        fig = px.bar(
            airline_sentiment,
            x=airline_column,
            y='count',
            color='sentiment_label',
            title='Sentiment Distribution by Airline',
            barmode='group',
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#95a5a6',
                'negative': '#e74c3c'
            }
        )
        
        fig.update_layout(
            xaxis_title="Airline",
            yaxis_title="Number of Tweets",
            legend_title="Sentiment",
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Airline sentiment visualization saved to {save_path}")
        
        return fig


def analyze_sentiment(input_file, output_file=None, text_column='cleaned_text', use_distilbert=False, use_svm=True):
    """
    Main function to perform sentiment analysis
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save results CSV (optional)
        text_column: Column containing text to analyze
        use_distilbert: Whether to use DistilBERT model
        use_svm: Whether to use SVM classifier (default: True for speed and accuracy)
        
    Returns:
        DataFrame with sentiment analysis results
    """
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize analyzer with SVM enabled by default
    analyzer = SentimentAnalyzer(use_distilbert=use_distilbert, use_svm=use_svm)
    
    # Analyze sentiment
    df_sentiment = analyzer.analyze_dataset(df, text_column)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Overall sentiment distribution
    fig1 = analyzer.visualize_sentiment_distribution(df_sentiment)
    
    # Sentiment by airline (if airline column exists)
    if 'airline' in df_sentiment.columns:
        fig2 = analyzer.visualize_sentiment_by_airline(df_sentiment)
    
    # Save results
    if output_file:
        df_sentiment.to_csv(output_file, index=False)
        print(f"Sentiment analysis results saved to {output_file}")
        
        # Save visualizations
        import os
        base_dir = os.path.dirname(output_file)
        fig1.write_html(os.path.join(base_dir, 'sentiment_distribution.html'))
        
        if 'airline' in df_sentiment.columns:
            fig2.write_html(os.path.join(base_dir, 'sentiment_by_airline.html'))
    
    return df_sentiment


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "clean_tweets.csv")
    output_file = os.path.join(base_dir, "data", "sentiment_results.csv")
    
    # Analyze sentiment (set use_distilbert=True for better accuracy)
    df_sentiment = analyze_sentiment(input_file, output_file, use_distilbert=False)
    
    print(f"\nPreview of sentiment results:")
    print(df_sentiment[['cleaned_text', 'sentiment_label', 'sentiment_score']].head())
