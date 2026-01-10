"""
Insights Generation Module for AI Narrative Nexus
Generates actionable insights from sentiment and topic analysis
"""

import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data silently
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

warnings.filterwarnings('ignore')


class InsightsGenerator:
    """
    Generate actionable insights from sentiment and topic analysis
    """
    
    def __init__(self, df, airline_column='airline'):
        """
        Initialize insights generator
        
        Args:
            df: DataFrame with sentiment and topic analysis results
            airline_column: Column containing airline names
        """
        self.df = df.copy()
        self.airline_column = airline_column
        self.insights = {}
        self.has_airline = airline_column in df.columns
        
        # Calculate text statistics
        self._calculate_text_statistics()
        
        # Extract keywords
        self._extract_keywords()
    
    def _calculate_text_statistics(self):
        """Calculate text length and word count statistics"""
        text_col = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'text'
        
        if text_col in self.df.columns:
            self.df['text_length'] = self.df[text_col].fillna('').str.len()
            self.df['word_count'] = self.df[text_col].fillna('').str.split().str.len()
        else:
            self.df['text_length'] = 0
            self.df['word_count'] = 0
    
    def _extract_keywords(self):
        """Extract keywords from text"""
        text_col = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'text'
        
        if text_col not in self.df.columns:
            self.df['keywords'] = [[] for _ in range(len(self.df))]
            return
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        def extract_keywords_from_text(text):
            if pd.isna(text) or not text:
                return []
            text = str(text).lower()
            # Remove URLs, emails, mentions
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            # Tokenize
            try:
                words = word_tokenize(text)
            except:
                words = text.split()
            # Filter
            keywords = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 2]
            return keywords
        
        self.df['keywords'] = self.df[text_col].apply(extract_keywords_from_text)
    
    def get_overall_statistics(self):
        """
        Get comprehensive overall statistics
        
        Returns:
            Dictionary with overall statistics
        """
        sentiment_col = 'sentiment_label' if 'sentiment_label' in self.df.columns else 'sentiment'
        
        # Sentiment distribution
        sentiment_counts = self.df[sentiment_col].value_counts()
        sentiment_percentages = (sentiment_counts / len(self.df) * 100).round(2)
        
        # Get all keywords
        all_keywords = []
        for keywords_list in self.df['keywords']:
            all_keywords.extend(keywords_list)
        
        keyword_freq = Counter(all_keywords).most_common(50)
        
        stats = {
            'total_records': len(self.df),
            'sentiment_counts': sentiment_counts.to_dict(),
            'sentiment_percentages': sentiment_percentages.to_dict(),
            'avg_text_length': round(self.df['text_length'].mean(), 1),
            'avg_word_count': round(self.df['word_count'].mean(), 1),
            'total_unique_keywords': len(set(all_keywords)),
            'top_keywords': keyword_freq[:30],
            'text_length_by_sentiment': self.df.groupby(sentiment_col)['text_length'].agg(['mean', 'median', 'std']).to_dict(),
            'word_count_by_sentiment': self.df.groupby(sentiment_col)['word_count'].agg(['mean', 'median', 'std']).to_dict()
        }
        
        if self.has_airline:
            stats['total_airlines'] = self.df[self.airline_column].nunique()
            stats['most_mentioned_airline'] = self.df[self.airline_column].value_counts().index[0]
            stats['airline_mentions'] = self.df[self.airline_column].value_counts().head(10).to_dict()
        
        self.insights['overall_statistics'] = stats
        return stats
    
    def get_keywords_by_sentiment(self):
        """
        Get top keywords for each sentiment category
        
        Returns:
            Dictionary with sentiment: [(keyword, count), ...]
        """
        sentiment_col = 'sentiment_label' if 'sentiment_label' in self.df.columns else 'sentiment'
        keywords_by_sentiment = {}
        
        for sentiment in self.df[sentiment_col].unique():
            sentiment_df = self.df[self.df[sentiment_col] == sentiment]
            sentiment_keywords = []
            for keywords_list in sentiment_df['keywords']:
                sentiment_keywords.extend(keywords_list)
            
            keyword_freq = Counter(sentiment_keywords).most_common(20)
            keywords_by_sentiment[sentiment] = keyword_freq
        
        self.insights['keywords_by_sentiment'] = keywords_by_sentiment
        return keywords_by_sentiment
        
    def analyze_sentiment_by_airline(self):
        """
        Analyze sentiment distribution by airline
        
        Returns:
            DataFrame with airline sentiment statistics
        """
        if not self.has_airline:
            return None
        
        # Group by airline and sentiment
        airline_sentiment = self.df.groupby([self.airline_column, 'sentiment_label']).size().unstack(fill_value=0)
        
        # Calculate percentages
        airline_sentiment_pct = airline_sentiment.div(airline_sentiment.sum(axis=1), axis=0) * 100
        
        # Calculate overall sentiment score
        sentiment_scores = self.df.groupby(self.airline_column)['sentiment_score'].agg(['mean', 'count'])
        
        # Combine
        airline_stats = pd.concat([airline_sentiment, airline_sentiment_pct, sentiment_scores], axis=1)
        airline_stats.columns = [
            'negative_count', 'neutral_count', 'positive_count',
            'negative_pct', 'neutral_pct', 'positive_pct',
            'avg_sentiment', 'total_mentions'
        ]
        
        # Rank airlines
        airline_stats['rank'] = airline_stats['avg_sentiment'].rank(ascending=False)
        airline_stats = airline_stats.sort_values('avg_sentiment', ascending=False)
        
        self.insights['airline_sentiment'] = airline_stats
        return airline_stats
    
    def identify_top_issues_by_airline(self):
        """
        Identify most common negative topics per airline
        
        Returns:
            Dictionary with airline: [top negative topics]
        """
        if not self.has_airline or 'topic_label' not in self.df.columns:
            return None
        
        # Filter negative sentiment
        negative_df = self.df[self.df['sentiment_label'] == 'negative']
        
        # Group by airline and topic
        issues = {}
        
        for airline in negative_df[self.airline_column].unique():
            airline_data = negative_df[negative_df[self.airline_column] == airline]
            top_topics = airline_data['topic_label'].value_counts().head(3)
            issues[airline] = top_topics.to_dict()
        
        self.insights['top_issues'] = issues
        return issues
    
    def identify_top_positive_aspects(self):
        """
        Identify most common positive topics per airline
        
        Returns:
            Dictionary with airline: [top positive topics]
        """
        if not self.has_airline or 'topic_label' not in self.df.columns:
            return None
        
        # Filter positive sentiment
        positive_df = self.df[self.df['sentiment_label'] == 'positive']
        
        # Group by airline and topic
        positives = {}
        
        for airline in positive_df[self.airline_column].unique():
            airline_data = positive_df[positive_df[self.airline_column] == airline]
            top_topics = airline_data['topic_label'].value_counts().head(3)
            positives[airline] = top_topics.to_dict()
        
        self.insights['top_positives'] = positives
        return positives
    
    def generate_word_clouds(self, output_dir=None):
        """
        Generate word clouds for each sentiment category
        
        Args:
            output_dir: Directory to save word cloud images
            
        Returns:
            Dictionary with sentiment: wordcloud object
        """
        word_clouds = {}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            # Get text for sentiment
            sentiment_text = ' '.join(
                self.df[self.df['sentiment_label'] == sentiment]['cleaned_text'].fillna('')
            )
            
            if len(sentiment_text.strip()) > 0:
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis' if sentiment == 'neutral' else ('Greens' if sentiment == 'positive' else 'Reds'),
                    max_words=100,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(sentiment_text)
                
                word_clouds[sentiment] = wordcloud
                
                # Save if output directory provided
                if output_dir:
                    import os
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'{sentiment.title()} Sentiment Word Cloud', fontsize=16, fontweight='bold')
                    plt.tight_layout(pad=0)
                    plt.savefig(os.path.join(output_dir, f'wordcloud_{sentiment}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        
        self.insights['word_clouds'] = word_clouds
        return word_clouds
    
    def generate_summary_text(self):
        """
        Generate comprehensive natural language summary of insights
        
        Returns:
            String with summary text
        """
        summary_lines = []
        
        # Get overall statistics if not already calculated
        if 'overall_statistics' not in self.insights:
            self.get_overall_statistics()
        
        stats = self.insights['overall_statistics']
        sentiment_col = 'sentiment_label' if 'sentiment_label' in self.df.columns else 'sentiment'
        
        # Header
        summary_lines.append("="*80)
        summary_lines.append("COMPREHENSIVE TEXT ANALYSIS - INSIGHTS SUMMARY")
        summary_lines.append("="*80)
        summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Total records analyzed: {stats['total_records']:,}")
        summary_lines.append("")
        
        # Dataset Overview
        summary_lines.append("📊 DATASET OVERVIEW")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Total records: {stats['total_records']:,}")
        if self.has_airline:
            summary_lines.append(f"  Airlines: {stats['total_airlines']}")
            summary_lines.append(f"  Most mentioned airline: {stats['most_mentioned_airline']}")
        summary_lines.append("")
        
        # Text Statistics
        summary_lines.append("📝 TEXT STATISTICS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Average text length: {stats['avg_text_length']:.1f} characters")
        summary_lines.append(f"  Average word count: {stats['avg_word_count']:.1f} words")
        summary_lines.append(f"  Total unique keywords: {stats['total_unique_keywords']:,}")
        summary_lines.append(f"  Top 5 keywords: {', '.join([w for w, _ in stats['top_keywords'][:5]])}")
        summary_lines.append("")
        
        # Overall sentiment distribution
        summary_lines.append("💭 SENTIMENT DISTRIBUTION")
        summary_lines.append("-" * 80)
        for sentiment, count in stats['sentiment_counts'].items():
            pct = stats['sentiment_percentages'][sentiment]
            summary_lines.append(f"  • {sentiment.upper():10s}: {count:6,} ({pct:.1f}%)")
        summary_lines.append("")
        
        # Top Keywords
        summary_lines.append("🔑 TOP 20 KEYWORDS (Overall)")
        summary_lines.append("-" * 80)
        for i, (word, count) in enumerate(stats['top_keywords'][:20], 1):
            summary_lines.append(f"  {i:2d}. {word:20s}: {count:5,} occurrences")
        summary_lines.append("")
        
        # Keywords by sentiment (if calculated)
        if 'keywords_by_sentiment' in self.insights:
            summary_lines.append("🎯 TOP KEYWORDS BY SENTIMENT")
            summary_lines.append("-" * 80)
            for sentiment, keywords in self.insights['keywords_by_sentiment'].items():
                summary_lines.append(f"{sentiment.upper()}:")
                for i, (word, count) in enumerate(keywords[:10], 1):
                    summary_lines.append(f"  {i:2d}. {word:20s}: {count:5,}")
                summary_lines.append("")
        
        # Airline mentions (if available)
        if self.has_airline and 'airline_mentions' in stats:
            summary_lines.append("✈️ AIRLINE MENTIONS")
            summary_lines.append("-" * 80)
            for airline, count in stats['airline_mentions'].items():
                pct = (count / stats['total_records']) * 100
                summary_lines.append(f"  • {airline:20s}: {count:6,} ({pct:.1f}%)")
            summary_lines.append("")
        
        # Airline rankings (if available)
        if self.has_airline and 'airline_sentiment' in self.insights:
            summary_lines.append("🏆 AIRLINE SENTIMENT RANKINGS")
            summary_lines.append("-" * 80)
            
            airline_stats = self.insights['airline_sentiment']
            
            # Best airline
            best_airline = airline_stats.index[0]
            best_score = airline_stats.iloc[0]['avg_sentiment']
            best_positive = airline_stats.iloc[0]['positive_pct']
            
            summary_lines.append(f"✅ BEST PERFORMING: {best_airline}")
            summary_lines.append(f"   Average sentiment score: {best_score:.4f}")
            summary_lines.append(f"   Positive mentions: {best_positive:.2f}%")
            summary_lines.append("")
            
            # Worst airline
            worst_airline = airline_stats.index[-1]
            worst_score = airline_stats.iloc[-1]['avg_sentiment']
            worst_negative = airline_stats.iloc[-1]['negative_pct']
            
            summary_lines.append(f"⚠️  NEEDS IMPROVEMENT: {worst_airline}")
            summary_lines.append(f"   Average sentiment score: {worst_score:.4f}")
            summary_lines.append(f"   Negative mentions: {worst_negative:.2f}%")
            summary_lines.append("")
            
            # All airlines
            summary_lines.append("All Airlines (ranked by sentiment):")
            for i, (airline, row) in enumerate(airline_stats.iterrows(), 1):
                summary_lines.append(
                    f"  {i}. {airline}: {row['avg_sentiment']:.4f} "
                    f"(+{row['positive_pct']:.1f}% / -{row['negative_pct']:.1f}%)"
                )
            summary_lines.append("")
        
        # Top issues (if available)
        if 'top_issues' in self.insights and self.insights['top_issues']:
            summary_lines.append("⚠️ MOST COMMON COMPLAINTS BY AIRLINE")
            summary_lines.append("-" * 80)
            
            for airline, topics in self.insights['top_issues'].items():
                summary_lines.append(f"{airline}:")
                for i, (topic, count) in enumerate(topics.items(), 1):
                    summary_lines.append(f"  {i}. {topic}: {count} mentions")
                summary_lines.append("")
        
        # Top positives (if available)
        if 'top_positives' in self.insights and self.insights['top_positives']:
            summary_lines.append("✨ MOST PRAISED ASPECTS BY AIRLINE")
            summary_lines.append("-" * 80)
            
            for airline, topics in self.insights['top_positives'].items():
                summary_lines.append(f"{airline}:")
                for i, (topic, count) in enumerate(topics.items(), 1):
                    summary_lines.append(f"  {i}. {topic}: {count} mentions")
                summary_lines.append("")
        
        # Topic distribution (if available)
        if 'topic_label' in self.df.columns:
            summary_lines.append("🏷️ TOPIC DISTRIBUTION")
            summary_lines.append("-" * 80)
            
            topic_counts = self.df['topic_label'].value_counts()
            for topic, count in topic_counts.items():
                pct = count / len(self.df) * 100
                summary_lines.append(f"  {topic}: {count:,} ({pct:.2f}%)")
            summary_lines.append("")
        
        # Recommendations
        summary_lines.append("💡 KEY RECOMMENDATIONS")
        summary_lines.append("-" * 80)
        
        if self.has_airline and 'airline_sentiment' in self.insights:
            airline_stats = self.insights['airline_sentiment']
            
            # For best airline
            best_airline = airline_stats.index[0]
            summary_lines.append(f"✅ {best_airline}:")
            summary_lines.append(f"   - Continue current customer service practices")
            summary_lines.append(f"   - Leverage positive feedback in marketing")
            summary_lines.append("")
            
            # For worst airline
            worst_airline = airline_stats.index[-1]
            if 'top_issues' in self.insights and worst_airline in self.insights['top_issues']:
                top_issue = list(self.insights['top_issues'][worst_airline].keys())[0]
                summary_lines.append(f"⚠️  {worst_airline}:")
                summary_lines.append(f"   - Priority: Address '{top_issue}' issues")
                summary_lines.append(f"   - Implement customer service training program")
                summary_lines.append(f"   - Increase communication during service disruptions")
                summary_lines.append("")
        
        summary_lines.append("="*80)
        summary_lines.append(f"📈 Visualizations Available: Sentiment charts, word clouds, topic analysis")
        summary_lines.append(f"📊 Interactive Charts: Airline comparisons, trend analysis, heatmaps")
        summary_lines.append("="*80)
        
        summary_text = '\n'.join(summary_lines)
        self.insights['summary_text'] = summary_text
        
        return summary_text
    
    def create_insights_visualizations(self, output_dir=None):
        """
        Create comprehensive insights visualizations
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with visualization figures
        """
        figures = {}
        sentiment_col = 'sentiment_label' if 'sentiment_label' in self.df.columns else 'sentiment'
        
        # Get overall statistics if not already calculated
        if 'overall_statistics' not in self.insights:
            self.get_overall_statistics()
        
        stats = self.insights['overall_statistics']
        
        # 1. Overall Sentiment Distribution (Bar Chart)
        sentiment_counts = pd.Series(stats['sentiment_counts'])
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'Sentiment', 'y': 'Count'},
            title='<b>Sentiment Distribution</b>',
            color=sentiment_counts.index,
            color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'},
            text=sentiment_counts.values
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=500)
        figures['sentiment_distribution'] = fig
        
        # 2. Sentiment Pie Chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='<b>Sentiment Distribution (Percentage)</b>',
            color=sentiment_counts.index,
            color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        figures['sentiment_pie'] = fig
        
        # 3. Top Keywords (Bar Chart)
        top_keywords = stats['top_keywords'][:20]
        words, counts = zip(*top_keywords) if top_keywords else ([], [])
        
        fig = px.bar(
            x=counts,
            y=words,
            orientation='h',
            title='<b>Top 20 Keywords</b>',
            labels={'x': 'Frequency', 'y': 'Keyword'},
            text=counts
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside', marker_color='steelblue')
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        figures['top_keywords'] = fig
        
        # 4. Text Length Distribution by Sentiment
        fig = px.histogram(
            self.df,
            x='text_length',
            color=sentiment_col,
            title='<b>Text Length Distribution by Sentiment</b>',
            labels={'text_length': 'Character Count', sentiment_col: 'Sentiment'},
            color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'},
            barmode='overlay',
            nbins=30
        )
        fig.update_traces(opacity=0.7)
        fig.update_layout(height=500)
        figures['text_length_distribution'] = fig
        
        # 5. Word Count Distribution by Sentiment
        fig = px.histogram(
            self.df,
            x='word_count',
            color=sentiment_col,
            title='<b>Word Count Distribution by Sentiment</b>',
            labels={'word_count': 'Word Count', sentiment_col: 'Sentiment'},
            color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'},
            barmode='overlay',
            nbins=30
        )
        fig.update_traces(opacity=0.7)
        fig.update_layout(height=500)
        figures['word_count_distribution'] = fig
        
        # 6. Box Plot - Text Length
        fig = px.box(
            self.df,
            x=sentiment_col,
            y='text_length',
            title='<b>Text Length Distribution by Sentiment (Box Plot)</b>',
            labels={'text_length': 'Character Count', sentiment_col: 'Sentiment'},
            color=sentiment_col,
            color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'}
        )
        fig.update_layout(showlegend=False, height=500)
        figures['text_length_box'] = fig
        
        # 7. Box Plot - Word Count
        fig = px.box(
            self.df,
            x=sentiment_col,
            y='word_count',
            title='<b>Word Count Distribution by Sentiment (Box Plot)</b>',
            labels={'word_count': 'Word Count', sentiment_col: 'Sentiment'},
            color=sentiment_col,
            color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'}
        )
        fig.update_layout(showlegend=False, height=500)
        figures['word_count_box'] = fig
        
        # 8. Keywords by Sentiment (Side-by-side comparison)
        if 'keywords_by_sentiment' in self.insights:
            keywords_data = []
            for sentiment, keywords in self.insights['keywords_by_sentiment'].items():
                for word, count in keywords[:10]:
                    keywords_data.append({'sentiment': sentiment, 'keyword': word, 'count': count})
            
            if keywords_data:
                keywords_df = pd.DataFrame(keywords_data)
                fig = px.bar(
                    keywords_df,
                    x='count',
                    y='keyword',
                    color='sentiment',
                    title='<b>Top 10 Keywords by Sentiment</b>',
                    labels={'count': 'Frequency', 'keyword': 'Keyword'},
                    color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'},
                    orientation='h',
                    barmode='group'
                )
                fig.update_layout(height=700)
                figures['keywords_by_sentiment'] = fig
        
        # 9. Airline-specific visualizations
        if self.has_airline and 'airline_sentiment' in self.insights:
            airline_stats = self.insights['airline_sentiment'].reset_index()
            
            # Stacked bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Positive',
                x=airline_stats[self.airline_column],
                y=airline_stats['positive_pct'],
                marker_color='#2ecc71',
                text=airline_stats['positive_pct'].round(1),
                texttemplate='%{text}%',
                textposition='inside'
            ))
            
            fig.add_trace(go.Bar(
                name='Neutral',
                x=airline_stats[self.airline_column],
                y=airline_stats['neutral_pct'],
                marker_color='#95a5a6',
                text=airline_stats['neutral_pct'].round(1),
                texttemplate='%{text}%',
                textposition='inside'
            ))
            
            fig.add_trace(go.Bar(
                name='Negative',
                x=airline_stats[self.airline_column],
                y=airline_stats['negative_pct'],
                marker_color='#e74c3c',
                text=airline_stats['negative_pct'].round(1),
                texttemplate='%{text}%',
                textposition='inside'
            ))
            
            fig.update_layout(
                title='<b>Sentiment Distribution by Airline (%)</b>',
                xaxis_title='Airline',
                yaxis_title='Percentage',
                barmode='stack',
                height=500,
                hovermode='x unified'
            )
            
            figures['airline_sentiment_stack'] = fig
            
            if output_dir:
                import os
                fig.write_html(os.path.join(output_dir, 'airline_comparison.html'))
        
        # 2. Sentiment score comparison
        if self.has_airline and 'airline_sentiment' in self.insights:
            airline_stats = self.insights['airline_sentiment'].reset_index()
            
            fig = px.bar(
                airline_stats,
                x=self.airline_column,
                y='avg_sentiment',
                title='<b>Average Sentiment Score by Airline</b>',
                color='avg_sentiment',
                color_continuous_scale=['red', 'yellow', 'green'],
                text='avg_sentiment',
                labels={'avg_sentiment': 'Avg Sentiment Score'}
            )
            
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(height=500)
            
            figures['airline_sentiment_scores'] = fig
            
            if output_dir:
                import os
                fig.write_html(os.path.join(output_dir, 'sentiment_scores.html'))
            
            # Sunburst chart
            fig = px.sunburst(
                self.df,
                path=[self.airline_column, sentiment_col],
                title='<b>Sentiment by Airline - Sunburst View</b>',
                color=sentiment_col,
                color_discrete_map={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'}
            )
            fig.update_layout(height=700)
            figures['airline_sunburst'] = fig
            
            # Treemap
            treemap_data = self.df.groupby([self.airline_column, sentiment_col]).size().reset_index(name='count')
            fig = px.treemap(
                treemap_data,
                path=[self.airline_column, sentiment_col],
                values='count',
                title='<b>Sentiment Distribution - Treemap View</b>',
                color='count',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=700)
            figures['airline_treemap'] = fig
            
            # Heatmap of sentiment percentages
            pivot_table = pd.crosstab(self.df[self.airline_column], self.df[sentiment_col], normalize='index') * 100
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdYlGn',
                text=pivot_table.values.round(1),
                texttemplate='%{text}%',
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='<b>Sentiment Distribution by Airline (%) - Heatmap</b>',
                xaxis_title='Sentiment',
                yaxis_title='Airline',
                height=600
            )
            
            figures['airline_heatmap'] = fig
        
        # Save all figures to output directory
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            for name, fig in figures.items():
                if hasattr(fig, 'write_html'):
                    fig.write_html(os.path.join(output_dir, f'{name}.html'))
        
        return figures
    
    def generate_all_insights(self, output_dir=None):
        """
        Generate all insights and save to output directory
        
        Args:
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary with all insights
        """
        print("🔍 Generating comprehensive insights...")
        
        # Calculate overall statistics
        print("📊 Calculating overall statistics...")
        self.get_overall_statistics()
        
        # Extract keywords by sentiment
        print("🔑 Extracting keywords by sentiment...")
        self.get_keywords_by_sentiment()
        
        # Analyze sentiment by airline
        if self.has_airline:
            print("✈️ Analyzing sentiment by airline...")
            self.analyze_sentiment_by_airline()
            
            print("⚠️ Identifying top issues...")
            self.identify_top_issues_by_airline()
            
            print("✨ Identifying positive aspects...")
            self.identify_top_positive_aspects()
        
        # Generate word clouds
        print("☁️ Generating word clouds...")
        self.generate_word_clouds(output_dir)
        
        # Generate summary text
        print("📝 Generating summary text...")
        summary = self.generate_summary_text()
        
        # Create visualizations
        print("📈 Creating visualizations...")
        figures = self.create_insights_visualizations(output_dir)
        
        # Save summary to file
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            summary_file = os.path.join(output_dir, 'insights_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"✅ Summary saved to {summary_file}")
            
            # Save airline statistics
            if 'airline_sentiment' in self.insights:
                stats_file = os.path.join(output_dir, 'airline_statistics.csv')
                self.insights['airline_sentiment'].to_csv(stats_file)
                print(f"✅ Airline statistics saved to {stats_file}")
            
            # Save overall statistics
            stats_dict = self.insights['overall_statistics'].copy()
            # Convert non-serializable items
            stats_dict['top_keywords'] = [{'keyword': k, 'count': c} for k, c in stats_dict['top_keywords']]
            
            import json
            stats_json_file = os.path.join(output_dir, 'overall_statistics.json')
            with open(stats_json_file, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, indent=2, default=str)
            print(f"✅ Overall statistics saved to {stats_json_file}")
        
        print("\n" + "="*80)
        print("✅ INSIGHTS GENERATION COMPLETE!")
        print("="*80)
        print(f"📊 Total visualizations created: {len(figures)}")
        print(f"📈 Summary text: {len(summary.split(chr(10)))} lines")
        print("="*80 + "\n")
        
        return self.insights


def generate_insights(input_file, output_dir=None, airline_column='airline'):
    """
    Main function to generate insights
    
    Args:
        input_file: Path to CSV with sentiment and topic results
        output_dir: Directory to save insights
        airline_column: Column containing airline names
        
    Returns:
        Dictionary with all insights
    """
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize generator
    generator = InsightsGenerator(df, airline_column)
    
    # Generate all insights
    insights = generator.generate_all_insights(output_dir)
    
    return insights


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "topic_results.csv")
    output_dir = os.path.join(base_dir, "reports")
    
    # Generate insights
    insights = generate_insights(input_file, output_dir)
