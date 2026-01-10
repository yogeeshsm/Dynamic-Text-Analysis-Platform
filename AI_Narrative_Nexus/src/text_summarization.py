"""
Text Summarization Module for AI Narrative Nexus
Performs extractive and abstractive summarization
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from collections import Counter
import re

class TextSummarizer:
    """
    Text summarization using extractive and abstractive methods
    """
    
    def __init__(self):
        """Initialize text summarizer (ULTRA-FAST)"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,  # Minimal features
            max_df=0.80,
            min_df=1
        )
    
    def extractive_summary(self, text, num_sentences=3):
        """
        Generate extractive summary by selecting most important sentences
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            Summary string
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate TF-IDF scores
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary = ' '.join([sentences[i] for i in top_indices])
            return summary
        except:
            # Fallback: return first num_sentences
            return ' '.join(sentences[:num_sentences])
    
    def abstractive_summary_simple(self, text, max_words=50):
        """
        Simple abstractive summary using word frequency and sentence compression
        
        Args:
            text: Input text to summarize
            max_words: Maximum words in summary
            
        Returns:
            Summary string
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # Clean and tokenize
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        # Get word frequencies
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'that', 'this', 'these',
                     'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'}
        
        # Get important words
        important_words = [(word, freq) for word, freq in word_freq.most_common(30) 
                          if word not in stop_words and len(word) > 2]
        
        # Extract key phrases from original sentences
        sentences = sent_tokenize(text)
        key_phrases = []
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            score = sum(freq for word, freq in important_words if word in sentence_words)
            if score > 0:
                key_phrases.append((sentence, score))
        
        # Sort by score and build summary
        key_phrases.sort(key=lambda x: x[1], reverse=True)
        
        summary_words = []
        word_count = 0
        
        for phrase, _ in key_phrases:
            phrase_words = phrase.split()
            if word_count + len(phrase_words) <= max_words:
                summary_words.extend(phrase_words)
                word_count += len(phrase_words)
            else:
                remaining = max_words - word_count
                summary_words.extend(phrase_words[:remaining])
                break
        
        return ' '.join(summary_words)


def summarize_dataset(df, text_column='cleaned_text', summary_type='extractive', 
                     num_sentences=3, max_words=50, batch_size=1000):
    """
    Generate summaries for dataset with batch processing for speed
    
    Args:
        df: DataFrame with text data
        text_column: Column containing text to summarize
        summary_type: 'extractive' or 'abstractive'
        num_sentences: Number of sentences for extractive summary
        max_words: Max words for abstractive summary
        batch_size: Process in batches for memory efficiency
        
    Returns:
        DataFrame with summaries added
    """
    summarizer = TextSummarizer()
    summaries = []
    
    # Process in batches for better performance
    total_rows = len(df)
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_texts = df[text_column].iloc[start_idx:end_idx]
        
        for text in batch_texts:
            if pd.isna(text):
                summaries.append("")
            else:
                if summary_type == 'extractive':
                    summary = summarizer.extractive_summary(str(text), num_sentences)
                else:
                    summary = summarizer.abstractive_summary_simple(str(text), max_words)
                summaries.append(summary)
    
    df[f'{summary_type}_summary'] = summaries
    return df


def generate_overall_summary(df, text_column='cleaned_text', sentiment_column='sentiment',
                            summary_type='extractive'):
    """
    Generate overall summary by sentiment category
    
    Args:
        df: DataFrame with text data
        text_column: Column containing text
        sentiment_column: Column containing sentiment labels
        summary_type: 'extractive' or 'abstractive'
        
    Returns:
        Dictionary with summaries by sentiment
    """
    summarizer = TextSummarizer()
    results = {}
    
    if sentiment_column in df.columns:
        for sentiment in df[sentiment_column].unique():
            sentiment_texts = df[df[sentiment_column] == sentiment][text_column]
            combined_text = ' '.join(sentiment_texts.dropna().astype(str).tolist())
            
            if summary_type == 'extractive':
                summary = summarizer.extractive_summary(combined_text, num_sentences=5)
            else:
                summary = summarizer.abstractive_summary_simple(combined_text, max_words=100)
            
            results[sentiment] = summary
    else:
        # No sentiment column, summarize all
        combined_text = ' '.join(df[text_column].dropna().astype(str).tolist())
        if summary_type == 'extractive':
            summary = summarizer.extractive_summary(combined_text, num_sentences=5)
        else:
            summary = summarizer.abstractive_summary_simple(combined_text, max_words=100)
        results['overall'] = summary
    
    return results


if __name__ == '__main__':
    # Example usage
    sample_text = """
    The flight was delayed for three hours. The customer service was terrible.
    Nobody informed us about the delay. The seats were uncomfortable.
    I will never fly with this airline again. The food was cold and tasteless.
    """
    
    summarizer = TextSummarizer()
    
    print("Original Text:")
    print(sample_text)
    print("\nExtractive Summary:")
    print(summarizer.extractive_summary(sample_text, 2))
    print("\nAbstractive Summary:")
    print(summarizer.abstractive_summary_simple(sample_text, 20))
