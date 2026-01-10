"""
Report Generator Module for AI Narrative Nexus
Creates comprehensive PDF reports with charts, topics, and insights
Enhanced with modern visualizations and interactive HTML reports
"""

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.platypus import Frame, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set enhanced plotting styles
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


class ReportGenerator:
    """
    Generate comprehensive PDF reports for sentiment analysis
    """
    
    def __init__(self, output_path, title="Airline Sentiment Analysis Report"):
        """
        Initialize report generator
        
        Args:
            output_path: Path to save PDF report
            title: Report title
        """
        self.output_path = output_path
        self.title = title
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Custom styles
        self.create_custom_styles()
    
    def create_custom_styles(self):
        """Create custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6,
            fontName='Helvetica'
        ))
        
        # Highlight style
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#27ae60'),
            fontName='Helvetica-Bold',
            spaceAfter=6
        ))
    
    def add_title_page(self):
        """Add title page to report"""
        # Title
        title = Paragraph(self.title, self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Subtitle
        subtitle = Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            self.styles['CustomBody']
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Description
        description = Paragraph(
            "This report provides a comprehensive analysis of airline sentiment based on Twitter data. "
            "It includes sentiment classification, topic modeling, and actionable insights to improve "
            "customer satisfaction and service quality.",
            self.styles['CustomBody']
        )
        self.story.append(description)
        self.story.append(PageBreak())
    
    def add_section_header(self, text):
        """Add section header"""
        header = Paragraph(text, self.styles['CustomHeading'])
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(header)
        self.story.append(Spacer(1, 0.1*inch))
    
    def add_subsection_header(self, text):
        """Add subsection header"""
        header = Paragraph(text, self.styles['CustomSubheading'])
        self.story.append(header)
    
    def add_paragraph(self, text, style='CustomBody'):
        """Add paragraph of text"""
        para = Paragraph(text, self.styles[style])
        self.story.append(para)
        self.story.append(Spacer(1, 0.1*inch))
    
    def add_summary_statistics(self, df):
        """Add summary statistics section"""
        self.add_section_header("1. Executive Summary")
        
        # Calculate statistics
        total_tweets = len(df)
        sentiment_counts = df['sentiment_label'].value_counts()
        avg_sentiment = df['sentiment_score'].mean()
        
        # Add statistics
        self.add_paragraph(f"<b>Total Tweets Analyzed:</b> {total_tweets:,}")
        self.add_paragraph(f"<b>Average Sentiment Score:</b> {avg_sentiment:.4f}")
        self.add_paragraph("")
        
        # Sentiment distribution
        self.add_subsection_header("Sentiment Distribution:")
        for sentiment in ['positive', 'neutral', 'negative']:
            count = sentiment_counts.get(sentiment, 0)
            pct = (count / total_tweets * 100) if total_tweets > 0 else 0
            self.add_paragraph(f"• {sentiment.title()}: {count:,} ({pct:.2f}%)")
    
    def add_table(self, data, col_widths=None):
        """Add table to report"""
        if col_widths is None:
            col_widths = [inch * (6.5 / len(data[0]))] * len(data[0])
        
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_airline_rankings(self, df):
        """Add airline rankings section"""
        if 'airline' not in df.columns:
            return
        
        self.add_section_header("2. Airline Performance Rankings")
        
        # Calculate airline statistics
        airline_stats = df.groupby('airline').agg({
            'sentiment_score': 'mean',
            'sentiment_label': lambda x: (x == 'positive').sum(),
            'text': 'count'
        }).reset_index()
        
        airline_stats.columns = ['Airline', 'Avg Sentiment', 'Positive Count', 'Total Mentions']
        airline_stats = airline_stats.sort_values('Avg Sentiment', ascending=False)
        airline_stats['Rank'] = range(1, len(airline_stats) + 1)
        
        # Reorder columns
        airline_stats = airline_stats[['Rank', 'Airline', 'Total Mentions', 'Positive Count', 'Avg Sentiment']]
        airline_stats['Avg Sentiment'] = airline_stats['Avg Sentiment'].round(4)
        
        # Create table
        table_data = [airline_stats.columns.tolist()] + airline_stats.values.tolist()
        self.add_table(table_data, col_widths=[0.7*inch, 1.5*inch, 1.3*inch, 1.3*inch, 1.2*inch])
    
    def add_topic_summary(self, topics_df):
        """Add topic modeling summary"""
        if topics_df is None or len(topics_df) == 0:
            return
        
        self.add_section_header("3. Topic Analysis")
        
        self.add_paragraph(
            "The following topics were identified through Latent Dirichlet Allocation (LDA) analysis:"
        )
        
        for _, row in topics_df.iterrows():
            self.add_subsection_header(f"Topic {row['topic_id'] + 1}: {row['topic_label']}")
            self.add_paragraph(f"<b>Keywords:</b> {row['keywords']}")
            self.add_paragraph("")
    
    def add_image_from_path(self, image_path, width=5*inch, height=3*inch):
        """Add image from file path"""
        if os.path.exists(image_path):
            img = Image(image_path, width=width, height=height)
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
        else:
            print(f"Warning: Image not found at {image_path}")
    
    def add_matplotlib_figure(self, fig, width=5*inch, height=3*inch):
        """Add matplotlib figure to report"""
        # Save figure to bytes buffer
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Add to report
        img = Image(img_buffer, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 0.2*inch))
        
        plt.close(fig)
    
    def add_text_summaries(self, summaries_dict):
        """Add text summaries section with extractive and abstractive summaries"""
        self.add_section_header("4. Text Summaries")
        
        # Add description
        self.add_paragraph(
            "This section presents automated summaries of the analyzed texts, "
            "organized by sentiment category. Both extractive (key sentences) and "
            "abstractive (generated condensed text) summaries are provided.",
            'Italic'
        )
        self.story.append(Spacer(1, 0.2*inch))
        
        # Check if we have extractive summaries
        if 'extractive' in summaries_dict:
            self.add_subsection_header("Extractive Summaries")
            self.add_paragraph(
                "Key sentences extracted from the original texts:",
                'Italic'
            )
            self.story.append(Spacer(1, 0.1*inch))
            
            for sentiment, summary in summaries_dict['extractive'].items():
                if summary:
                    # Color code by sentiment
                    color = '#2ecc71' if sentiment == 'positive' else '#e74c3c' if sentiment == 'negative' else '#95a5a6'
                    
                    self.add_paragraph(f"<b>{sentiment.upper()}:</b>", 'Heading3')
                    
                    # Create colored box for summary
                    summary_style = ParagraphStyle(
                        'SummaryBox',
                        parent=self.styles['Normal'],
                        fontSize=9,
                        leading=12,
                        leftIndent=20,
                        rightIndent=20,
                        spaceBefore=6,
                        spaceAfter=6,
                        textColor=colors.HexColor('#333333'),
                        backColor=colors.HexColor('#f8f9fa'),
                        borderColor=colors.HexColor(color),
                        borderWidth=2,
                        borderPadding=10
                    )
                    
                    summary_para = Paragraph(summary[:800] + "..." if len(summary) > 800 else summary, summary_style)
                    self.story.append(summary_para)
                    self.story.append(Spacer(1, 0.15*inch))
        
        # Check if we have abstractive summaries
        if 'abstractive' in summaries_dict:
            self.story.append(Spacer(1, 0.2*inch))
            self.add_subsection_header("Abstractive Summaries")
            self.add_paragraph(
                "Condensed summaries generated from key phrases and words:",
                'Italic'
            )
            self.story.append(Spacer(1, 0.1*inch))
            
            for sentiment, summary in summaries_dict['abstractive'].items():
                if summary:
                    # Color code by sentiment
                    color = '#2ecc71' if sentiment == 'positive' else '#e74c3c' if sentiment == 'negative' else '#95a5a6'
                    
                    self.add_paragraph(f"<b>{sentiment.upper()}:</b>", 'Heading3')
                    
                    # Create colored box for summary
                    summary_style = ParagraphStyle(
                        'SummaryBox',
                        parent=self.styles['Normal'],
                        fontSize=9,
                        leading=12,
                        leftIndent=20,
                        rightIndent=20,
                        spaceBefore=6,
                        spaceAfter=6,
                        textColor=colors.HexColor('#333333'),
                        backColor=colors.HexColor('#f8f9fa'),
                        borderColor=colors.HexColor(color),
                        borderWidth=2,
                        borderPadding=10
                    )
                    
                    summary_para = Paragraph(summary[:600] + "..." if len(summary) > 600 else summary, summary_style)
                    self.story.append(summary_para)
                    self.story.append(Spacer(1, 0.15*inch))
    
    def add_key_insights(self, insights_text):
        """Add key insights section"""
        self.add_section_header("5. Key Insights & Recommendations")
        
        # Parse insights text
        lines = insights_text.split('\n')
        
        for line in lines:
            if line.strip():
                if line.startswith('✅') or line.startswith('⚠️'):
                    self.add_paragraph(f"<b>{line}</b>", 'Highlight')
                elif line.startswith('  -'):
                    self.add_paragraph(f"   {line.strip()}")
                elif not line.startswith('=') and not line.startswith('-'):
                    self.add_paragraph(line)
    
    def add_wordcloud_images(self, wordcloud_dir):
        """Add word cloud images"""
        self.add_section_header("6. Word Cloud Visualizations")
        
        for sentiment in ['positive', 'negative', 'neutral']:
            img_path = os.path.join(wordcloud_dir, f'wordcloud_{sentiment}.png')
            if os.path.exists(img_path):
                self.add_subsection_header(f"{sentiment.title()} Sentiment Word Cloud")
                self.add_image_from_path(img_path, width=5.5*inch, height=2.75*inch)
    
    def build(self):
        """Build the PDF report"""
        try:
            self.doc.build(self.story)
            print(f"Report successfully generated: {self.output_path}")
            return True
        except Exception as e:
            print(f"Error generating report: {e}")
            return False


def generate_interactive_dashboard(df, topics_df=None, insights_text=None, output_path=None):
    """
    Generate comprehensive interactive HTML dashboard with modern visualizations
    
    Args:
        df: DataFrame with sentiment analysis results
        topics_df: DataFrame with topic information
        insights_text: Text summary of insights
        output_path: Path to save HTML dashboard
        
    Returns:
        Path to generated dashboard
    """
    if output_path is None:
        output_path = "interactive_dashboard.html"
    
    print(f"Generating interactive dashboard: {output_path}")
    
    # Create subplot layout
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Sentiment Distribution', 'Airline Performance',
            'Sentiment Trends by Airline', 'Topic Distribution',
            'Sentiment Score Distribution', 'Word Count vs Sentiment',
            'Text Length Analysis', 'Airline Comparison Matrix'
        ),
        specs=[
            [{'type': 'pie'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'pie'}],
            [{'type': 'histogram'}, {'type': 'scatter'}],
            [{'type': 'box'}, {'type': 'heatmap'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Sentiment Distribution Pie Chart
    sentiment_counts = df['sentiment_label'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker=dict(
                colors=['#2ecc71', '#95a5a6', '#e74c3c'],
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=12, color='white')
        ),
        row=1, col=1
    )
    
    # 2. Airline Performance Bar Chart
    if 'airline' in df.columns:
        airline_sentiment = df.groupby('airline')['sentiment_score'].mean().sort_values()
        colors_bar = ['#e74c3c' if x < -0.1 else '#2ecc71' if x > 0.1 else '#f39c12' 
                      for x in airline_sentiment.values]
        
        fig.add_trace(
            go.Bar(
                x=airline_sentiment.values,
                y=airline_sentiment.index,
                orientation='h',
                marker=dict(color=colors_bar, line=dict(color='white', width=1)),
                text=[f'{x:.3f}' for x in airline_sentiment.values],
                textposition='auto'
            ),
            row=1, col=2
        )
    
    # 3. Sentiment Trends by Airline (Stacked Bar)
    if 'airline' in df.columns:
        airline_dist = df.groupby(['airline', 'sentiment_label']).size().unstack(fill_value=0)
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in airline_dist.columns:
                color = '#2ecc71' if sentiment == 'positive' else '#95a5a6' if sentiment == 'neutral' else '#e74c3c'
                fig.add_trace(
                    go.Bar(
                        name=sentiment.title(),
                        x=airline_dist.index,
                        y=airline_dist[sentiment],
                        marker=dict(color=color, line=dict(color='white', width=1)),
                        showlegend=True
                    ),
                    row=2, col=1
                )
    
    # 4. Topic Distribution
    if topics_df is not None and 'topic_label' in df.columns:
        topic_counts = df['topic_label'].value_counts().head(8)
        fig.add_trace(
            go.Pie(
                labels=topic_counts.index,
                values=topic_counts.values,
                hole=0.3,
                marker=dict(line=dict(color='white', width=2)),
                textinfo='label+percent'
            ),
            row=2, col=2
        )
    
    # 5. Sentiment Score Distribution (Histogram)
    fig.add_trace(
        go.Histogram(
            x=df['sentiment_score'],
            nbinsx=50,
            marker=dict(
                color=df['sentiment_score'],
                colorscale='RdYlGn',
                line=dict(color='white', width=1)
            ),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 6. Word Count vs Sentiment (Scatter)
    if 'word_count' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['word_count'],
                y=df['sentiment_score'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df['sentiment_score'],
                    colorscale='RdYlGn',
                    line=dict(width=0.5, color='white'),
                    opacity=0.6
                ),
                showlegend=False
            ),
            row=3, col=2
        )
    
    # 7. Text Length Box Plot
    if 'text_length' in df.columns:
        for sentiment in df['sentiment_label'].unique():
            color = '#2ecc71' if sentiment == 'positive' else '#95a5a6' if sentiment == 'neutral' else '#e74c3c'
            sentiment_data = df[df['sentiment_label'] == sentiment]['text_length']
            
            fig.add_trace(
                go.Box(
                    y=sentiment_data,
                    name=sentiment.title(),
                    marker=dict(color=color),
                    boxmean='sd'
                ),
                row=4, col=1
            )
    
    # 8. Airline Comparison Heatmap
    if 'airline' in df.columns:
        # Create airline vs sentiment matrix
        heatmap_data = df.groupby(['airline', 'sentiment_label']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100  # Convert to percentages
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                text=np.round(heatmap_data.values, 1),
                texttemplate='%{text}%',
                textfont={"size": 10},
                showscale=True
            ),
            row=4, col=2
        )
    
    # Update layout with modern styling
    fig.update_layout(
        title=dict(
            text='<b>AI Narrative Nexus - Comprehensive Sentiment Analysis Dashboard</b>',
            font=dict(size=24, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        height=1800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11, color='#2c3e50'),
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    
    # Save dashboard
    fig.write_html(output_path)
    print(f"✅ Interactive dashboard generated: {output_path}")
    
    # Generate additional modern visualizations
    base_dir = os.path.dirname(output_path)
    
    print("\n🎨 Generating additional modern visualizations...")
    generate_sunburst_chart(df, os.path.join(base_dir, 'airline_sunburst.html'))
    generate_treemap_chart(df, os.path.join(base_dir, 'airline_treemap.html'))
    generate_3d_scatter(df, os.path.join(base_dir, 'sentiment_3d_scatter.html'))
    generate_radial_chart(df, os.path.join(base_dir, 'airline_radial.html'))
    generate_sankey_diagram(df, os.path.join(base_dir, 'sentiment_sankey.html'))
    
    print(f"\n✅ All visualizations generated successfully!")
    
    return output_path


def generate_sunburst_chart(df, output_path=None):
    """
    Generate hierarchical sunburst chart for airline sentiment breakdown
    
    Args:
        df: DataFrame with sentiment analysis results
        output_path: Path to save HTML file
        
    Returns:
        Path to generated file
    """
    if output_path is None:
        output_path = "airline_sunburst.html"
    
    print(f"Generating sunburst chart: {output_path}")
    
    if 'airline' not in df.columns:
        print("Airline column not found, skipping sunburst chart")
        return None
    
    # Create hierarchical data
    hierarchy_data = df.groupby(['airline', 'sentiment_label']).size().reset_index(name='count')
    
    # Create sunburst
    fig = px.sunburst(
        hierarchy_data,
        path=['airline', 'sentiment_label'],
        values='count',
        color='sentiment_label',
        color_discrete_map={
            'positive': '#2ecc71',
            'neutral': '#f39c12',
            'negative': '#e74c3c'
        },
        title='<b>Airline Sentiment Hierarchy - Sunburst View</b>',
        hover_data=['count']
    )
    
    fig.update_layout(
        font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
        paper_bgcolor='#f8f9fa',
        height=600,
        template='plotly_white'
    )
    
    fig.write_html(output_path)
    print(f"✅ Sunburst chart saved: {output_path}")
    return output_path


def generate_treemap_chart(df, output_path=None):
    """
    Generate treemap visualization for sentiment distribution by airline
    
    Args:
        df: DataFrame with sentiment analysis results
        output_path: Path to save HTML file
        
    Returns:
        Path to generated file
    """
    if output_path is None:
        output_path = "airline_treemap.html"
    
    print(f"Generating treemap chart: {output_path}")
    
    if 'airline' not in df.columns:
        print("Airline column not found, skipping treemap")
        return None
    
    # Create hierarchical data
    treemap_data = df.groupby(['airline', 'sentiment_label']).size().reset_index(name='count')
    treemap_data['percentage'] = treemap_data.groupby('airline')['count'].transform(lambda x: x / x.sum() * 100)
    
    # Create treemap with custom colors
    fig = px.treemap(
        treemap_data,
        path=['airline', 'sentiment_label'],
        values='count',
        color='sentiment_label',
        color_discrete_map={
            'positive': '#27ae60',
            'neutral': '#f39c12',
            'negative': '#c0392b'
        },
        title='<b>Sentiment Distribution by Airline - Treemap View</b>',
        hover_data={'count': True, 'percentage': ':.1f'}
    )
    
    fig.update_layout(
        font=dict(family='Arial, sans-serif', size=12, color='white'),
        paper_bgcolor='#f8f9fa',
        height=600,
        template='plotly_white'
    )
    
    fig.update_traces(
        textfont=dict(size=14, color='white'),
        marker=dict(line=dict(width=2, color='white'))
    )
    
    fig.write_html(output_path)
    print(f"✅ Treemap saved: {output_path}")
    return output_path


def generate_3d_scatter(df, output_path=None):
    """
    Generate 3D scatter plot for sentiment analysis with word count and text length
    
    Args:
        df: DataFrame with sentiment analysis results
        output_path: Path to save HTML file
        
    Returns:
        Path to generated file
    """
    if output_path is None:
        output_path = "sentiment_3d_scatter.html"
    
    print(f"Generating 3D scatter plot: {output_path}")
    
    # Add computed columns if missing
    if 'word_count' not in df.columns and 'cleaned_text' in df.columns:
        df['word_count'] = df['cleaned_text'].fillna('').str.split().str.len()
    
    if 'text_length' not in df.columns and 'cleaned_text' in df.columns:
        df['text_length'] = df['cleaned_text'].fillna('').str.len()
    
    # Sample data for performance (max 5000 points)
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    
    # Create 3D scatter
    fig = px.scatter_3d(
        sample_df,
        x='word_count' if 'word_count' in sample_df.columns else sample_df.index,
        y='text_length' if 'text_length' in sample_df.columns else sample_df.index,
        z='sentiment_score',
        color='sentiment_label',
        color_discrete_map={
            'positive': '#2ecc71',
            'neutral': '#95a5a6',
            'negative': '#e74c3c'
        },
        title='<b>3D Sentiment Analysis - Word Count vs Text Length vs Score</b>',
        labels={
            'word_count': 'Word Count',
            'text_length': 'Text Length',
            'sentiment_score': 'Sentiment Score'
        },
        opacity=0.7,
        size_max=8
    )
    
    fig.update_layout(
        font=dict(family='Arial, sans-serif', size=11, color='#2c3e50'),
        paper_bgcolor='#f8f9fa',
        height=700,
        template='plotly_white',
        scene=dict(
            xaxis=dict(backgroundcolor='#ecf0f1', gridcolor='white'),
            yaxis=dict(backgroundcolor='#ecf0f1', gridcolor='white'),
            zaxis=dict(backgroundcolor='#ecf0f1', gridcolor='white')
        )
    )
    
    fig.write_html(output_path)
    print(f"✅ 3D scatter plot saved: {output_path}")
    return output_path


def generate_radial_chart(df, output_path=None):
    """
    Generate radial/polar bar chart for airline sentiment comparison
    
    Args:
        df: DataFrame with sentiment analysis results
        output_path: Path to save HTML file
        
    Returns:
        Path to generated file
    """
    if output_path is None:
        output_path = "airline_radial.html"
    
    print(f"Generating radial chart: {output_path}")
    
    if 'airline' not in df.columns:
        print("Airline column not found, skipping radial chart")
        return None
    
    # Calculate average sentiment by airline
    airline_stats = df.groupby('airline').agg({
        'sentiment_score': 'mean'
    }).reset_index()
    airline_stats['count'] = df.groupby('airline').size().values
    
    # Normalize to 0-100 scale for better visualization
    airline_stats['normalized_score'] = (airline_stats['sentiment_score'] + 1) * 50
    
    # Create radial bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Barpolar(
        r=airline_stats['normalized_score'],
        theta=airline_stats['airline'],
        marker=dict(
            color=airline_stats['sentiment_score'],
            colorscale='RdYlGn',
            line=dict(color='white', width=2),
            cmin=-1,
            cmax=1
        ),
        text=[f'{score:.3f}' for score in airline_stats['sentiment_score']],
        hovertemplate='<b>%{theta}</b><br>Sentiment: %{text}<br>Count: %{customdata}<extra></extra>',
        customdata=airline_stats['count']
    ))
    
    fig.update_layout(
        title='<b>Airline Sentiment Comparison - Radial View</b>',
        font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            ),
            bgcolor='#f8f9fa'
        ),
        showlegend=False,
        height=600,
        paper_bgcolor='#f8f9fa',
        template='plotly_white'
    )
    
    fig.write_html(output_path)
    print(f"✅ Radial chart saved: {output_path}")
    return output_path


def generate_sankey_diagram(df, output_path=None):
    """
    Generate Sankey diagram showing flow from airlines to sentiments to topics
    
    Args:
        df: DataFrame with sentiment analysis results
        output_path: Path to save HTML file
        
    Returns:
        Path to generated file
    """
    if output_path is None:
        output_path = "sentiment_sankey.html"
    
    print(f"Generating Sankey diagram: {output_path}")
    
    if 'airline' not in df.columns:
        print("Airline column not found, skipping Sankey diagram")
        return None
    
    # Create flow data
    airlines = df['airline'].unique().tolist()
    sentiments = df['sentiment_label'].unique().tolist()
    
    # Create nodes
    all_nodes = airlines + sentiments
    node_colors = ['#3498db'] * len(airlines) + [
        '#2ecc71' if s == 'positive' else '#f39c12' if s == 'neutral' else '#e74c3c'
        for s in sentiments
    ]
    
    # Create links
    flows = df.groupby(['airline', 'sentiment_label']).size().reset_index(name='count')
    
    source_indices = [all_nodes.index(airline) for airline in flows['airline']]
    target_indices = [all_nodes.index(sentiment) for sentiment in flows['sentiment_label']]
    values = flows['count'].tolist()
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=2),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color='rgba(0,0,0,0.2)'
        )
    )])
    
    fig.update_layout(
        title='<b>Sentiment Flow by Airline - Sankey Diagram</b>',
        font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
        height=600,
        paper_bgcolor='#f8f9fa',
        template='plotly_white'
    )
    
    fig.write_html(output_path)
    print(f"✅ Sankey diagram saved: {output_path}")
    return output_path


def generate_pdf_report(df, topics_df=None, insights_text=None, 
                       output_path=None, wordcloud_dir=None, summaries_dict=None):
    """
    Generate comprehensive PDF report
    
    Args:
        df: DataFrame with analysis results
        topics_df: DataFrame with topic information
        insights_text: Text summary of insights
        output_path: Path to save PDF
        wordcloud_dir: Directory containing word cloud images
        summaries_dict: Dictionary with extractive and abstractive summaries by sentiment
        
    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = "airline_sentiment_report.pdf"
    
    print(f"Generating PDF report: {output_path}")
    
    # Create report generator
    report = ReportGenerator(output_path)
    
    # Add title page
    report.add_title_page()
    
    # Add summary statistics
    report.add_summary_statistics(df)
    report.story.append(PageBreak())
    
    # Add airline rankings
    report.add_airline_rankings(df)
    report.story.append(PageBreak())
    
    # Add topic summary
    if topics_df is not None:
        report.add_topic_summary(topics_df)
        report.story.append(PageBreak())
    
    # Add text summaries section
    if summaries_dict:
        report.add_text_summaries(summaries_dict)
        report.story.append(PageBreak())
    
    # Add key insights
    if insights_text:
        report.add_key_insights(insights_text)
        report.story.append(PageBreak())
    
    # Add word clouds
    if wordcloud_dir and os.path.exists(wordcloud_dir):
        report.add_wordcloud_images(wordcloud_dir)
    
    # Build report
    success = report.build()
    
    return success


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "topic_results.csv")
    topics_file = os.path.join(base_dir, "data", "topics.csv")
    insights_file = os.path.join(base_dir, "reports", "insights_summary.txt")
    output_path = os.path.join(base_dir, "reports", "airline_sentiment_report.pdf")
    wordcloud_dir = os.path.join(base_dir, "reports")
    
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
    
    # Generate report
    generate_pdf_report(
        df,
        topics_df=topics_df,
        insights_text=insights_text,
        output_path=output_path,
        wordcloud_dir=wordcloud_dir
    )
