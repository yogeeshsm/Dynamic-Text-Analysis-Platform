"""
Topic Modeling Module for AI Narrative Nexus
Performs topic extraction using LDA and NMF
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim not available. Using sklearn-based topic modeling only.")
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class TopicModeler:
    """
    Topic modeling using LDA and NMF algorithms
    """
    
    def __init__(self, n_topics=5, method='lda', random_state=42):
        """
        Initialize topic modeler
        
        Args:
            n_topics: Number of topics to extract
            method: 'lda' or 'nmf'
            random_state: Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.method = method.lower()
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.topic_labels = []
        
    def prepare_documents(self, df, text_column='cleaned_text'):
        """
        Prepare documents for topic modeling
        
        Args:
            df: DataFrame with text data
            text_column: Column containing cleaned text
            
        Returns:
            List of documents
        """
        # Auto-detect text column if default not found
        if text_column not in df.columns:
            for col in ['text', 'cleaned_text', 'processed_text', 'content']:
                if col in df.columns:
                    text_column = col
                    print(f"Using column '{text_column}' for topic modeling")
                    break
            else:
                raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")
        
        documents = df[text_column].fillna('').tolist()
        # Filter out empty documents
        documents = [doc for doc in documents if len(doc.strip()) > 0]
        return documents
    
    def create_document_term_matrix(self, documents):
        """
        Create document-term matrix
        
        Args:
            documents: List of text documents
            
        Returns:
            Document-term matrix and vectorizer
        """
        # ULTRA-FAST: minimal features but ensure enough for analysis
        n_docs = len(documents)
        
        # Ensure we have enough documents
        if n_docs < 2:
            raise ValueError(f"Need at least 2 documents for topic modeling, got {n_docs}")
        
        # Smarter feature limits based on dataset size
        if n_docs < 50:
            min_df = 1
            max_features = min(50, max(10, n_docs))
        elif n_docs < 100:
            min_df = 1
            max_features = min(75, max(15, n_docs))
        else:
            min_df = 1 if n_docs < 10 else (2 if n_docs < 50 else 3)
            max_features = min(100, max(20, n_docs // 10))
        
        if self.method == 'lda':
            # Use CountVectorizer for LDA
            self.vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=min_df,
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 1),  # Only unigrams for small datasets
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # At least 2 chars
            )
        else:
            # Use TfidfVectorizer for NMF
            self.vectorizer = TfidfVectorizer(
                max_df=0.95,
                min_df=min_df,
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 1),  # Only unigrams for small datasets
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # At least 2 chars
            )
        
        try:
            dtm = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Ensure we have features
            if dtm.shape[1] == 0:
                raise ValueError("No features extracted from documents")
                
        except (ValueError, AttributeError) as e:
            print(f"Warning: {e}. Using fallback settings...")
            # If still no terms, use most lenient settings
            if self.method == 'lda':
                self.vectorizer = CountVectorizer(
                    max_df=1.0,
                    min_df=1,
                    max_features=50,
                    ngram_range=(1, 1),
                    token_pattern=r'\b[a-zA-Z]+\b'
                )
            else:
                self.vectorizer = TfidfVectorizer(
                    max_df=1.0,
                    min_df=1,
                    max_features=50,
                    ngram_range=(1, 1),
                    token_pattern=r'\b[a-zA-Z]+\b'
                )
            dtm = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            if dtm.shape[1] == 0:
                raise ValueError("Unable to extract any features from documents. Check if texts are properly cleaned.")
        
        print(f"Created DTM: {dtm.shape[0]} documents, {dtm.shape[1]} features")
        return dtm
    
    def train_model(self, dtm):
        """
        Train topic model
        
        Args:
            dtm: Document-term matrix
            
        Returns:
            Trained model
        """
        # Adjust number of topics based on data size
        n_docs = dtm.shape[0]
        n_features = dtm.shape[1]
        
        # Smart topic adjustment for small datasets
        if n_docs < 50:
            max_topics = max(2, n_docs // 10)  # Very small: 2-4 topics max
        elif n_docs < 100:
            max_topics = max(2, n_docs // 15)  # Small: 2-6 topics max
        elif n_docs < 500:
            max_topics = max(3, n_docs // 50)  # Medium: 3-10 topics max
        else:
            max_topics = self.n_topics  # Large: use requested topics
        
        # Can't have more topics than documents or features
        adjusted_topics = min(self.n_topics, max_topics, n_docs - 1, n_features)
        adjusted_topics = max(2, adjusted_topics)  # At least 2 topics for analysis
        
        # Update n_topics to reflect actual number
        self.n_topics = adjusted_topics
        
        print(f"Adjusted topics from requested to {adjusted_topics} based on dataset size ({n_docs} docs, {n_features} features)")
        
        # EXTREME SPEED MODE: <10 second target
        max_iter_lda = 1  # Single iteration
        max_iter_nmf = 10  # Minimal iterations
        perplexity_check = -1
        
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=adjusted_topics,
                random_state=self.random_state,
                max_iter=max_iter_lda,
                learning_method='online',
                learning_decay=0.5,  # Fastest convergence
                learning_offset=20.0,  # Very fast initial learning
                batch_size=1024,  # Maximum batch size
                n_jobs=-1,
                evaluate_every=perplexity_check,
                verbose=0
            )
        else:
            self.model = NMF(
                n_components=adjusted_topics,
                random_state=self.random_state,
                max_iter=max_iter_nmf,
                init='random',  # Fastest initialization
                solver='cd',
                beta_loss='frobenius',
                tol=0.01,  # Very loose tolerance
                alpha_W=0.0,
                alpha_H=0.0,
                l1_ratio=0.0
            )
        
        self.model.fit(dtm)
        return self.model
    
    def get_top_words(self, topic_idx, n_words=10):
        """
        Get top words for a topic
        
        Args:
            topic_idx: Topic index
            n_words: Number of top words to return
            
        Returns:
            List of (word, weight) tuples
        """
        topic = self.model.components_[topic_idx]
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [(self.feature_names[i], topic[i]) for i in top_indices]
        return top_words
    
    def generate_topic_labels(self):
        """
        Generate human-readable labels for topics based on top keywords
        
        Returns:
            List of topic labels
        """
        topic_labels = []
        
        # Predefined topic categories for airline tweets
        airline_topics = {
            'delay flight late time': 'Flight Delays',
            'service customer bad terrible': 'Customer Service Issues',
            'bag baggage luggage lost': 'Baggage Problems',
            'cancelled cancel flight booking': 'Flight Cancellations',
            'seat plane comfort space': 'Seat & Comfort',
            'staff crew rude helpful': 'Staff Behavior',
            'price ticket cheap expensive': 'Pricing & Value',
            'food drink meal snack': 'Food & Beverage',
            'gate boarding queue line': 'Boarding Issues',
            'thank great awesome love': 'Positive Experience'
        }
        
        for i in range(self.n_topics):
            top_words = self.get_top_words(i, n_words=10)
            word_list = ' '.join([word for word, _ in top_words])
            
            # Try to match with predefined categories
            label = f"Topic {i+1}"
            max_match = 0
            
            for keywords, category in airline_topics.items():
                matches = sum(1 for kw in keywords.split() if kw in word_list)
                if matches > max_match:
                    max_match = matches
                    label = category
            
            # If no good match, use top 3 words
            if max_match < 2:
                top_3_words = [word for word, _ in top_words[:3]]
                label = ' & '.join(top_3_words).title()
            
            topic_labels.append(label)
        
        self.topic_labels = topic_labels
        return topic_labels
    
    def get_document_topics(self, dtm):
        """
        Get topic distribution for each document
        
        Args:
            dtm: Document-term matrix
            
        Returns:
            Array of topic distributions
        """
        doc_topics = self.model.transform(dtm)
        return doc_topics
    
    def assign_dominant_topics(self, doc_topics):
        """
        Assign dominant topic to each document
        
        Args:
            doc_topics: Topic distributions for documents
            
        Returns:
            Array of dominant topic indices
        """
        # Handle edge case: if doc_topics has only 1 or 2 topics but trying to access more
        if doc_topics.shape[1] == 0:
            # No topics available
            return np.zeros(doc_topics.shape[0], dtype=int)
        elif doc_topics.shape[1] == 1:
            # Only one topic, all documents get topic 0
            return np.zeros(doc_topics.shape[0], dtype=int)
        else:
            # Normal case: find the dominant topic
            dominant_topics = doc_topics.argmax(axis=1)
            return dominant_topics
    
    def analyze_topics(self, df, text_column='cleaned_text'):
        """
        Complete topic modeling pipeline
        
        Args:
            df: DataFrame with text data
            text_column: Column containing cleaned text
            
        Returns:
            DataFrame with topic assignments and topic information
        """
        print(f"Starting topic modeling with {self.method.upper()}...")
        print(f"Number of topics: {self.n_topics}")
        
        # Prepare documents
        print("Preparing documents...")
        documents = self.prepare_documents(df, text_column)
        print(f"Total documents: {len(documents)}")
        
        # Create document-term matrix
        print("Creating document-term matrix...")
        dtm = self.create_document_term_matrix(documents)
        print(f"Vocabulary size: {len(self.feature_names)}")
        
        # Train model
        print(f"Training {self.method.upper()} model...")
        self.train_model(dtm)
        
        # Generate topic labels
        print("Generating topic labels...")
        self.generate_topic_labels()
        
        # Get document topics
        print("Assigning topics to documents...")
        doc_topics = self.get_document_topics(dtm)
        dominant_topics = self.assign_dominant_topics(doc_topics)
        
        # Create results dataframe
        df_topics = df.copy()
        df_topics['dominant_topic'] = dominant_topics
        
        # Safely assign topic labels with bounds checking
        topic_labels_safe = []
        for topic_idx in dominant_topics:
            if 0 <= topic_idx < len(self.topic_labels):
                topic_labels_safe.append(self.topic_labels[topic_idx])
            else:
                topic_labels_safe.append(f"Topic {topic_idx}")
        df_topics['topic_label'] = topic_labels_safe
        
        # Add topic probabilities with bounds checking
        for i in range(min(self.n_topics, doc_topics.shape[1])):
            df_topics[f'topic_{i}_prob'] = doc_topics[:, i]
        
        # Print topic summary
        print("\n" + "="*60)
        print("TOPIC MODELING RESULTS")
        print("="*60)
        
        actual_topics = min(self.n_topics, len(self.topic_labels), doc_topics.shape[1])
        for i in range(actual_topics):
            top_words = self.get_top_words(i, n_words=10)
            word_str = ', '.join([word for word, _ in top_words[:10]])
            count = (dominant_topics == i).sum()
            percentage = count / len(dominant_topics) * 100 if len(dominant_topics) > 0 else 0
            
            topic_label = self.topic_labels[i] if i < len(self.topic_labels) else f"Topic {i}"
            print(f"\n{topic_label} (Topic {i}):")
            print(f"  Documents: {count} ({percentage:.2f}%)")
            print(f"  Keywords: {word_str}")
        
        print("\n" + "="*60 + "\n")
        
        return df_topics
    
    def get_topics_dataframe(self, n_words=10):
        """
        Create a DataFrame with topic information
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            DataFrame with topic details
        """
        topics_data = []
        
        for i in range(self.n_topics):
            top_words = self.get_top_words(i, n_words)
            
            topics_data.append({
                'topic_id': i,
                'topic_label': self.topic_labels[i],
                'keywords': ', '.join([word for word, _ in top_words]),
                'top_words': [word for word, _ in top_words],
                'weights': [float(weight) for _, weight in top_words]
            })
        
        return pd.DataFrame(topics_data)
    
    def visualize_topics(self, df_topics, save_path=None):
        """
        Create visualization of topic distribution
        
        Args:
            df_topics: DataFrame with topic assignments
            save_path: Path to save visualization
            
        Returns:
            Plotly figure
        """
        # Count topics
        topic_counts = df_topics['topic_label'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        
        # Create bar chart
        fig = px.bar(
            topic_counts,
            x='Topic',
            y='Count',
            title=f'Topic Distribution ({self.method.upper()})',
            color='Count',
            color_continuous_scale='Viridis',
            text='Count'
        )
        
        fig.update_layout(
            xaxis_title="Topic",
            yaxis_title="Number of Documents",
            height=500,
            showlegend=False
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Topic visualization saved to {save_path}")
        
        return fig
    
    def visualize_topic_keywords(self, n_words=10, save_path=None):
        """
        Create visualization of top keywords per topic
        
        Args:
            n_words: Number of words to show per topic
            save_path: Path to save visualization
            
        Returns:
            Plotly figure
        """
        # Prepare data
        data = []
        for i in range(self.n_topics):
            top_words = self.get_top_words(i, n_words)
            for word, weight in top_words:
                data.append({
                    'Topic': self.topic_labels[i],
                    'Word': word,
                    'Weight': weight
                })
        
        df_viz = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig = px.bar(
            df_viz,
            x='Weight',
            y='Word',
            color='Topic',
            orientation='h',
            title='Top Keywords by Topic',
            facet_col='Topic',
            facet_col_wrap=2,
            height=300 * ((self.n_topics + 1) // 2)
        )
        
        fig.update_yaxes(matches=None, showticklabels=True)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Keyword visualization saved to {save_path}")
        
        return fig


def perform_topic_modeling(input_file, output_file=None, n_topics=5, 
                          method='lda', text_column='cleaned_text'):
    """
    Main function to perform topic modeling
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save results CSV
        n_topics: Number of topics to extract
        method: 'lda' or 'nmf'
        text_column: Column containing cleaned text
        
    Returns:
        DataFrame with topic assignments and topic information DataFrame
    """
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize modeler
    modeler = TopicModeler(n_topics=n_topics, method=method)
    
    # Analyze topics
    df_topics = modeler.analyze_topics(df, text_column)
    
    # Get topic information
    topics_df = modeler.get_topics_dataframe(n_words=10)
    
    # Create visualizations
    print("Creating visualizations...")
    fig1 = modeler.visualize_topics(df_topics)
    fig2 = modeler.visualize_topic_keywords(n_words=8)
    
    # Save results
    if output_file:
        df_topics.to_csv(output_file, index=False)
        print(f"Topic modeling results saved to {output_file}")
        
        # Save topics information
        import os
        base_dir = os.path.dirname(output_file)
        topics_file = os.path.join(base_dir, 'topics.csv')
        topics_df.to_csv(topics_file, index=False)
        print(f"Topics information saved to {topics_file}")
        
        # Save visualizations
        fig1.write_html(os.path.join(base_dir, 'topic_distribution.html'))
        fig2.write_html(os.path.join(base_dir, 'topic_keywords.html'))
    
    return df_topics, topics_df


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "sentiment_results.csv")
    output_file = os.path.join(base_dir, "data", "topic_results.csv")
    
    # Perform topic modeling
    df_topics, topics_df = perform_topic_modeling(
        input_file, 
        output_file, 
        n_topics=7,
        method='lda'
    )
    
    print(f"\nPreview of topic assignments:")
    print(df_topics[['cleaned_text', 'topic_label', 'dominant_topic']].head())
    
    print(f"\nTopics summary:")
    print(topics_df[['topic_label', 'keywords']])
