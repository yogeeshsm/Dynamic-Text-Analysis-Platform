"""
Data Processing Module for AI Narrative Nexus
Handles text preprocessing: cleaning, tokenization, lemmatization
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for tweet data
    """
    
    def __init__(self):
        """Initialize preprocessor and download required NLTK data"""
        self._download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Pre-compile regex patterns for speed
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
            "]+", flags=re.UNICODE)
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        if pd.isna(text):
            return ""
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text):
        """Remove @mentions from text"""
        if pd.isna(text):
            return ""
        return self.mention_pattern.sub('', text)
    
    def remove_hashtags(self, text):
        """Remove # from hashtags but keep the text"""
        if pd.isna(text):
            return ""
        return self.hashtag_pattern.sub('', text)
    
    def remove_emojis(self, text):
        """Remove emojis and special unicode characters"""
        if pd.isna(text):
            return ""
        return self.emoji_pattern.sub(r'', text)
    
    def remove_special_chars(self, text):
        """Remove special characters and numbers, keep only letters"""
        if pd.isna(text):
            return ""
        # Keep only alphabetic characters and spaces
        text = self.special_char_pattern.sub('', text)
        # Remove extra whitespaces
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def to_lowercase(self, text):
        """Convert text to lowercase"""
        if pd.isna(text):
            return ""
        return text.lower()
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        if pd.isna(text) or text == "":
            return []
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        if not tokens:
            return []
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens (EXTREME SPEED: disabled for speed)"""
        # Skip lemmatization entirely for maximum speed
        return tokens
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Step 1: Remove URLs
        text = self.remove_urls(text)
        
        # Step 2: Remove mentions
        text = self.remove_mentions(text)
        
        # Step 3: Remove hashtag symbols
        text = self.remove_hashtags(text)
        
        # Step 4: Remove emojis
        text = self.remove_emojis(text)
        
        # Step 5: Convert to lowercase
        text = self.to_lowercase(text)
        
        # Step 6: Remove special characters
        text = self.remove_special_chars(text)
        
        # Step 7: Tokenize
        tokens = self.tokenize_text(text)
        
        # Step 8: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 9: Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def process_dataset(self, df, text_column='text'):
        """
        Process entire dataset
        
        Args:
            df: DataFrame containing tweets
            text_column: Name of column containing tweet text
            
        Returns:
            DataFrame with cleaned text and preprocessing stats
        """
        print("Starting text preprocessing...")
        print(f"Total tweets: {len(df)}")
        
        # Create a copy
        df_clean = df.copy()
        
        # Store original text
        df_clean['original_text'] = df_clean[text_column]
        
        # Apply preprocessing
        print("Cleaning text...")
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.preprocess_text)
        
        # Calculate word counts
        df_clean['word_count_original'] = df_clean['original_text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        df_clean['word_count_cleaned'] = df_clean['cleaned_text'].apply(
            lambda x: len(x.split()) if x else 0
        )
        
        # Remove empty tweets after cleaning
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
        removed_count = initial_count - len(df_clean)
        
        print(f"Removed {removed_count} empty tweets after cleaning")
        print(f"Final tweet count: {len(df_clean)}")
        
        return df_clean
    
    def generate_preprocessing_summary(self, df_original, df_cleaned):
        """
        Generate summary statistics of preprocessing
        
        Args:
            df_original: Original DataFrame
            df_cleaned: Cleaned DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_tweets_original': len(df_original),
            'total_tweets_cleaned': len(df_cleaned),
            'tweets_removed': len(df_original) - len(df_cleaned),
            'avg_words_original': df_cleaned['word_count_original'].mean(),
            'avg_words_cleaned': df_cleaned['word_count_cleaned'].mean(),
            'total_words_removed': df_cleaned['word_count_original'].sum() - df_cleaned['word_count_cleaned'].sum(),
            'reduction_percentage': ((df_cleaned['word_count_original'].sum() - df_cleaned['word_count_cleaned'].sum()) / 
                                    df_cleaned['word_count_original'].sum() * 100)
        }
        
        return summary


def load_and_preprocess_data(input_file, output_file=None, text_column='text'):
    """
    Main function to load and preprocess airline sentiment data
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save cleaned CSV (optional)
        text_column: Name of column containing tweet text
        
    Returns:
        Cleaned DataFrame and preprocessing summary
    """
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Process dataset
    df_cleaned = preprocessor.process_dataset(df, text_column)
    
    # Generate summary
    summary = preprocessor.generate_preprocessing_summary(df, df_cleaned)
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original tweets: {summary['total_tweets_original']:,}")
    print(f"Cleaned tweets: {summary['total_tweets_cleaned']:,}")
    print(f"Tweets removed: {summary['tweets_removed']:,}")
    print(f"Avg words (original): {summary['avg_words_original']:.2f}")
    print(f"Avg words (cleaned): {summary['avg_words_cleaned']:.2f}")
    print(f"Total words removed: {summary['total_words_removed']:,}")
    print(f"Word reduction: {summary['reduction_percentage']:.2f}%")
    print("="*60 + "\n")
    
    # Save cleaned data
    if output_file:
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
        # Save summary report
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("PREPROCESSING SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        print(f"Summary report saved to {summary_file}")
    
    return df_cleaned, summary


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "cleaned_dataset.csv.csv")
    output_file = os.path.join(base_dir, "data", "clean_tweets.csv")
    
    # Process data
    df_cleaned, summary = load_and_preprocess_data(input_file, output_file, text_column='text')
    
    print(f"\nPreview of cleaned data:")
    print(df_cleaned[['original_text', 'cleaned_text']].head())
