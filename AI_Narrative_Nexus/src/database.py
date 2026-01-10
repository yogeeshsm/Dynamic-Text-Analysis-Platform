"""
Database Module for AI Narrative Nexus
Manages SQLite database for storing all processed data and analysis results
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class AnalysisDatabase:
    """
    SQLite database manager for storing analysis data
    """
    
    def __init__(self, db_path='data/analysis.db'):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _get_connection(self):
        """
        Get database connection with optimized settings to prevent locking
        
        Returns:
            sqlite3.Connection: Database connection with timeout and WAL mode
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False, isolation_level=None)
        # Enable WAL mode for better concurrency (allows multiple readers during writes)
        conn.execute('PRAGMA journal_mode=WAL')
        # Reduce busy timeout
        conn.execute('PRAGMA busy_timeout=30000')
        return conn
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Table 1: Analysis Sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT,
                total_records INTEGER,
                source_file TEXT,
                description TEXT
            )
        ''')
        
        # Table 2: Raw Data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                text TEXT,
                airline TEXT,
                airline_sentiment TEXT,
                original_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
            )
        ''')
        
        # Table 3: Preprocessed Data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preprocessed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                raw_id INTEGER,
                cleaned_text TEXT,
                word_count INTEGER,
                char_count INTEGER,
                has_urls BOOLEAN,
                has_mentions BOOLEAN,
                has_hashtags BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id),
                FOREIGN KEY (raw_id) REFERENCES raw_data(id)
            )
        ''')
        
        # Table 4: Sentiment Results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                raw_id INTEGER,
                preprocessed_id INTEGER,
                sentiment_label TEXT,
                sentiment_score REAL,
                vader_positive REAL,
                vader_negative REAL,
                vader_neutral REAL,
                vader_compound REAL,
                textblob_polarity REAL,
                textblob_subjectivity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id),
                FOREIGN KEY (raw_id) REFERENCES raw_data(id),
                FOREIGN KEY (preprocessed_id) REFERENCES preprocessed_data(id)
            )
        ''')
        
        # Table 5: Topic Results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                raw_id INTEGER,
                topic_id INTEGER,
                topic_label TEXT,
                topic_probability REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id),
                FOREIGN KEY (raw_id) REFERENCES raw_data(id)
            )
        ''')
        
        # Table 6: Topic Definitions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_definitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                topic_id INTEGER,
                topic_label TEXT,
                top_words TEXT,
                word_weights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
            )
        ''')
        
        # Table 7: Insights
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                insight_type TEXT,
                airline TEXT,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
            )
        ''')
        
        # Table 8: Keywords
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                keyword TEXT,
                frequency INTEGER,
                sentiment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
            )
        ''')
        
        # Table 9: Analysis Statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                stat_category TEXT,
                stat_name TEXT,
                stat_value REAL,
                stat_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_session ON raw_data(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_preprocessed_session ON preprocessed_data(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_session ON sentiment_results(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_session ON topic_results(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_insights_session ON insights(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords_session ON keywords(session_id)')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def create_session(self, dataset_name, total_records, source_file='', description=''):
        """
        Create a new analysis session
        
        Args:
            dataset_name: Name of the dataset
            total_records: Number of records in dataset
            source_file: Source file path
            description: Session description
            
        Returns:
            session_id: ID of created session
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_sessions (dataset_name, total_records, source_file, description)
            VALUES (?, ?, ?, ?)
        ''', (dataset_name, total_records, source_file, description))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Created analysis session #{session_id}")
        return session_id
    
    def save_raw_data(self, session_id, df):
        """
        Save raw data to database
        
        Args:
            session_id: Session ID
            df: DataFrame with raw data
        """
        conn = self._get_connection()
        
        # Prepare data
        data_to_insert = []
        for _, row in df.iterrows():
            # Store all original columns as JSON
            original_data = row.to_dict()
            original_data_json = json.dumps(original_data, default=str)
            
            data_to_insert.append((
                session_id,
                row.get('text', row.get('clean_text', '')),
                row.get('airline', ''),
                row.get('airline_sentiment', row.get('sentiment', '')),
                original_data_json
            ))
        
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT INTO raw_data (session_id, text, airline, airline_sentiment, original_data)
            VALUES (?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {len(data_to_insert)} raw records to database")
    
    def save_preprocessed_data(self, session_id, df):
        """
        Save preprocessed data to database
        
        Args:
            session_id: Session ID
            df: DataFrame with preprocessed data
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get raw_ids in order
        cursor.execute('SELECT id FROM raw_data WHERE session_id = ? ORDER BY id', (session_id,))
        raw_ids = [row[0] for row in cursor.fetchall()]
        
        data_to_insert = []
        for idx, row in df.iterrows():
            if idx < len(raw_ids):
                raw_id = raw_ids[idx]
                
                text = row.get('cleaned_text', row.get('text', ''))
                data_to_insert.append((
                    session_id,
                    raw_id,
                    text,
                    row.get('word_count', len(str(text).split())),
                    row.get('char_count', len(str(text))),
                    row.get('has_urls', False),
                    row.get('has_mentions', False),
                    row.get('has_hashtags', False)
                ))
        
        cursor.executemany('''
            INSERT INTO preprocessed_data 
            (session_id, raw_id, cleaned_text, word_count, char_count, has_urls, has_mentions, has_hashtags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {len(data_to_insert)} preprocessed records to database")
    
    def save_sentiment_results(self, session_id, df):
        """
        Save sentiment analysis results to database
        
        Args:
            session_id: Session ID
            df: DataFrame with sentiment results
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get raw_ids and preprocessed_ids
        cursor.execute('SELECT id FROM raw_data WHERE session_id = ? ORDER BY id', (session_id,))
        raw_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.execute('SELECT id FROM preprocessed_data WHERE session_id = ? ORDER BY id', (session_id,))
        preprocessed_ids = [row[0] for row in cursor.fetchall()]
        
        data_to_insert = []
        for idx, row in df.iterrows():
            if idx < len(raw_ids):
                data_to_insert.append((
                    session_id,
                    raw_ids[idx],
                    preprocessed_ids[idx] if idx < len(preprocessed_ids) else None,
                    row.get('sentiment_label', row.get('sentiment', '')),
                    row.get('sentiment_score', 0.0),
                    row.get('vader_positive', row.get('positive', 0.0)),
                    row.get('vader_negative', row.get('negative', 0.0)),
                    row.get('vader_neutral', row.get('neutral', 0.0)),
                    row.get('vader_compound', row.get('compound', 0.0)),
                    row.get('textblob_polarity', 0.0),
                    row.get('textblob_subjectivity', 0.0)
                ))
        
        cursor.executemany('''
            INSERT INTO sentiment_results 
            (session_id, raw_id, preprocessed_id, sentiment_label, sentiment_score,
             vader_positive, vader_negative, vader_neutral, vader_compound,
             textblob_polarity, textblob_subjectivity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {len(data_to_insert)} sentiment results to database")
    
    def save_topic_results(self, session_id, df, topics_df=None):
        """
        Save topic modeling results to database
        
        Args:
            session_id: Session ID
            df: DataFrame with topic assignments
            topics_df: DataFrame with topic definitions
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get raw_ids
        cursor.execute('SELECT id FROM raw_data WHERE session_id = ? ORDER BY id', (session_id,))
        raw_ids = [row[0] for row in cursor.fetchall()]
        
        # Save topic assignments
        data_to_insert = []
        for idx, row in df.iterrows():
            if idx < len(raw_ids):
                # Ensure all values are proper types for SQLite
                topic_id = row.get('topic_id', row.get('topic', row.get('dominant_topic', 0)))
                topic_label = row.get('topic_label', '')
                topic_prob = row.get('topic_probability', row.get('probability', 0.0))
                
                # Convert to appropriate types
                topic_id = int(topic_id) if topic_id is not None else 0
                topic_label = str(topic_label) if topic_label is not None else ''
                topic_prob = float(topic_prob) if topic_prob is not None else 0.0
                
                data_to_insert.append((
                    session_id,
                    raw_ids[idx],
                    topic_id,
                    topic_label,
                    topic_prob
                ))
        
        cursor.executemany('''
            INSERT INTO topic_results (session_id, raw_id, topic_id, topic_label, topic_probability)
            VALUES (?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        print(f"✅ Saved {len(data_to_insert)} topic assignments to database")
        
        # Save topic definitions if provided
        if topics_df is not None:
            topic_defs = []
            for _, row in topics_df.iterrows():
                # Convert top_words to string if it's a list
                top_words = row.get('top_words', '')
                if isinstance(top_words, list):
                    top_words = ', '.join(str(w) for w in top_words)
                
                # Convert word_weights to JSON if it's a dict or list
                word_weights = row.get('word_weights', {})
                if not isinstance(word_weights, str):
                    word_weights = json.dumps(word_weights, default=str)
                
                topic_defs.append((
                    session_id,
                    int(row.get('topic_id', row.get('topic', 0))),
                    str(row.get('topic_label', '')),
                    str(top_words),
                    str(word_weights)
                ))
            
            cursor.executemany('''
                INSERT INTO topic_definitions (session_id, topic_id, topic_label, top_words, word_weights)
                VALUES (?, ?, ?, ?, ?)
            ''', topic_defs)
            
            print(f"✅ Saved {len(topic_defs)} topic definitions to database")
        
        conn.commit()
        conn.close()
    
    def save_insights(self, session_id, insights_dict):
        """
        Save insights to database
        
        Args:
            session_id: Session ID
            insights_dict: Dictionary with insights data
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Save overall statistics
        if 'overall_statistics' in insights_dict:
            stats = insights_dict['overall_statistics']
            
            stats_to_insert = [
                (session_id, 'overall', 'total_records', stats.get('total_records', 0), ''),
                (session_id, 'overall', 'avg_text_length', stats.get('avg_text_length', 0.0), ''),
                (session_id, 'overall', 'avg_word_count', stats.get('avg_word_count', 0.0), ''),
                (session_id, 'overall', 'total_unique_keywords', stats.get('total_unique_keywords', 0), ''),
            ]
            
            # Add sentiment percentages
            if 'sentiment_percentages' in stats:
                for sentiment, pct in stats['sentiment_percentages'].items():
                    stats_to_insert.append((
                        session_id, 'sentiment_pct', sentiment, pct, ''
                    ))
            
            cursor.executemany('''
                INSERT INTO statistics (session_id, stat_category, stat_name, stat_value, stat_data)
                VALUES (?, ?, ?, ?, ?)
            ''', stats_to_insert)
            
            print(f"✅ Saved {len(stats_to_insert)} statistics to database")
        
        # Save keywords
        if 'overall_statistics' in insights_dict and 'top_keywords' in insights_dict['overall_statistics']:
            keywords_to_insert = []
            for keyword, count in insights_dict['overall_statistics']['top_keywords']:
                keywords_to_insert.append((session_id, keyword, count, 'overall'))
            
            cursor.executemany('''
                INSERT INTO keywords (session_id, keyword, frequency, sentiment)
                VALUES (?, ?, ?, ?)
            ''', keywords_to_insert)
            
            print(f"✅ Saved {len(keywords_to_insert)} keywords to database")
        
        # Save keywords by sentiment
        if 'keywords_by_sentiment' in insights_dict:
            keywords_to_insert = []
            for sentiment, keywords in insights_dict['keywords_by_sentiment'].items():
                for keyword, count in keywords:
                    keywords_to_insert.append((session_id, keyword, count, sentiment))
            
            cursor.executemany('''
                INSERT INTO keywords (session_id, keyword, frequency, sentiment)
                VALUES (?, ?, ?, ?)
            ''', keywords_to_insert)
            
            print(f"✅ Saved {len(keywords_to_insert)} sentiment-specific keywords to database")
        
        # Save airline insights
        if 'airline_sentiment' in insights_dict:
            airline_df = insights_dict['airline_sentiment']
            insights_to_insert = []
            
            for airline, row in airline_df.iterrows():
                insights_to_insert.extend([
                    (session_id, 'airline_sentiment', airline, 'avg_sentiment', row['avg_sentiment'], ''),
                    (session_id, 'airline_sentiment', airline, 'positive_pct', row['positive_pct'], ''),
                    (session_id, 'airline_sentiment', airline, 'negative_pct', row['negative_pct'], ''),
                    (session_id, 'airline_sentiment', airline, 'neutral_pct', row['neutral_pct'], ''),
                    (session_id, 'airline_sentiment', airline, 'total_mentions', row['total_mentions'], ''),
                ])
            
            cursor.executemany('''
                INSERT INTO insights (session_id, insight_type, airline, metric_name, metric_value, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', insights_to_insert)
            
            print(f"✅ Saved {len(insights_to_insert)} airline insights to database")
        
        conn.commit()
        conn.close()
    
    def get_session_data(self, session_id):
        """
        Get all data for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with all session data
        """
        conn = self._get_connection()
        
        # Get session info
        session_df = pd.read_sql_query(
            'SELECT * FROM analysis_sessions WHERE session_id = ?',
            conn, params=(session_id,)
        )
        
        # Get combined data
        combined_df = pd.read_sql_query('''
            SELECT 
                r.id,
                r.text as original_text,
                r.airline,
                r.airline_sentiment as original_sentiment,
                p.cleaned_text,
                p.word_count,
                p.char_count,
                s.sentiment_label,
                s.sentiment_score,
                s.vader_compound,
                t.topic_id,
                t.topic_label,
                t.topic_probability
            FROM raw_data r
            LEFT JOIN preprocessed_data p ON r.id = p.raw_id
            LEFT JOIN sentiment_results s ON r.id = s.raw_id
            LEFT JOIN topic_results t ON r.id = t.raw_id
            WHERE r.session_id = ?
            ORDER BY r.id
        ''', conn, params=(session_id,))
        
        conn.close()
        
        return {
            'session_info': session_df.to_dict('records')[0] if not session_df.empty else {},
            'data': combined_df
        }
    
    def list_sessions(self):
        """
        List all analysis sessions
        
        Returns:
            DataFrame with session information
        """
        conn = self._get_connection()
        df = pd.read_sql_query(
            'SELECT * FROM analysis_sessions ORDER BY created_at DESC',
            conn
        )
        conn.close()
        return df
    
    def get_latest_session_id(self):
        """Get the ID of the most recent session"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(session_id) FROM analysis_sessions')
        session_id = cursor.fetchone()[0]
        conn.close()
        return session_id
    
    def delete_session(self, session_id):
        """
        Delete a session and all its data
        
        Args:
            session_id: Session ID to delete
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete in reverse order of foreign keys
        cursor.execute('DELETE FROM insights WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM keywords WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM statistics WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM topic_definitions WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM topic_results WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sentiment_results WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM preprocessed_data WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM raw_data WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM analysis_sessions WHERE session_id = ?', (session_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted session #{session_id}")


# Convenience function
def get_database(db_path='data/analysis.db'):
    """Get database instance"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, db_path)
    return AnalysisDatabase(full_path)


if __name__ == "__main__":
    # Test database creation
    db = get_database()
    sessions = db.list_sessions()
    print(f"\n📊 Total sessions: {len(sessions)}")
    if not sessions.empty:
        print(sessions)
