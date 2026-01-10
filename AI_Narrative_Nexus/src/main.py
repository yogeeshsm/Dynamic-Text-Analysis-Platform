"""
Main Pipeline Orchestrator for AI Narrative Nexus
Executes the complete text analysis workflow
"""

import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data_processing import load_and_preprocess_data
from sentiment_analysis import analyze_sentiment
from topic_modeling import perform_topic_modeling
from insights_generation import generate_insights
from report_generator import generate_pdf_report
import pandas as pd


class TextAnalysisPipeline:
    """
    Orchestrates the complete text analysis pipeline
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize pipeline
        
        Args:
            base_dir: Base directory for the project
        """
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
        
        # Define paths
        self.data_dir = os.path.join(os.path.dirname(self.base_dir), "data")
        self.reports_dir = os.path.join(os.path.dirname(self.base_dir), "reports")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # File paths
        self.raw_data_path = os.path.join(self.data_dir, "cleaned_dataset.csv.csv")
        self.clean_data_path = os.path.join(self.data_dir, "clean_tweets.csv")
        self.sentiment_data_path = os.path.join(self.data_dir, "sentiment_results.csv")
        self.topic_data_path = os.path.join(self.data_dir, "topic_results.csv")
        self.topics_info_path = os.path.join(self.data_dir, "topics.csv")
        self.insights_path = os.path.join(self.reports_dir, "insights_summary.txt")
        self.report_path = os.path.join(self.reports_dir, "airline_sentiment_report.pdf")
        
        # Configuration
        self.config = {
            'text_column': 'text',
            'airline_column': 'airline',
            'use_distilbert': False,
            'n_topics': 7,
            'topic_method': 'lda'
        }
        
        self.results = {}
    
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70 + "\n")
    
    def run_preprocessing(self):
        """Step 1: Data Preprocessing"""
        self.print_header("STEP 1: DATA PREPROCESSING")
        
        if not os.path.exists(self.raw_data_path):
            print(f"Error: Raw data file not found at {self.raw_data_path}")
            print("Please ensure cleaned_dataset.csv.csv is in the data/ directory")
            return False
        
        try:
            df_clean, summary = load_and_preprocess_data(
                self.raw_data_path,
                self.clean_data_path,
                self.config['text_column']
            )
            
            self.results['preprocessing'] = {
                'data': df_clean,
                'summary': summary
            }
            
            print(f"✓ Preprocessing completed successfully!")
            print(f"  Cleaned data saved to: {self.clean_data_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_sentiment_analysis(self):
        """Step 2: Sentiment Analysis"""
        self.print_header("STEP 2: SENTIMENT ANALYSIS")
        
        if not os.path.exists(self.clean_data_path):
            print(f"Error: Cleaned data not found. Run preprocessing first.")
            return False
        
        try:
            df_sentiment = analyze_sentiment(
                self.clean_data_path,
                self.sentiment_data_path,
                text_column='cleaned_text',
                use_distilbert=self.config['use_distilbert']
            )
            
            self.results['sentiment'] = df_sentiment
            
            print(f"✓ Sentiment analysis completed successfully!")
            print(f"  Results saved to: {self.sentiment_data_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error during sentiment analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_topic_modeling(self):
        """Step 3: Topic Modeling"""
        self.print_header("STEP 3: TOPIC MODELING")
        
        if not os.path.exists(self.sentiment_data_path):
            print(f"Error: Sentiment data not found. Run sentiment analysis first.")
            return False
        
        try:
            df_topics, topics_df = perform_topic_modeling(
                self.sentiment_data_path,
                self.topic_data_path,
                n_topics=self.config['n_topics'],
                method=self.config['topic_method'],
                text_column='cleaned_text'
            )
            
            self.results['topics'] = {
                'data': df_topics,
                'topics_info': topics_df
            }
            
            print(f"✓ Topic modeling completed successfully!")
            print(f"  Results saved to: {self.topic_data_path}")
            print(f"  Topic info saved to: {self.topics_info_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error during topic modeling: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_insights_generation(self):
        """Step 4: Insights Generation"""
        self.print_header("STEP 4: INSIGHTS GENERATION")
        
        if not os.path.exists(self.topic_data_path):
            print(f"Error: Topic data not found. Run topic modeling first.")
            return False
        
        try:
            insights = generate_insights(
                self.topic_data_path,
                self.reports_dir,
                self.config['airline_column']
            )
            
            self.results['insights'] = insights
            
            print(f"✓ Insights generation completed successfully!")
            print(f"  Insights saved to: {self.insights_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error during insights generation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_report_generation(self):
        """Step 5: Report Generation"""
        self.print_header("STEP 5: REPORT GENERATION")
        
        try:
            # Load data
            df = pd.read_csv(self.topic_data_path)
            
            # Load topics info
            topics_df = None
            if os.path.exists(self.topics_info_path):
                topics_df = pd.read_csv(self.topics_info_path)
            
            # Load insights text
            insights_text = None
            if os.path.exists(self.insights_path):
                with open(self.insights_path, 'r', encoding='utf-8') as f:
                    insights_text = f.read()
            
            # Generate report
            success = generate_pdf_report(
                df,
                topics_df=topics_df,
                insights_text=insights_text,
                output_path=self.report_path,
                wordcloud_dir=self.reports_dir
            )
            
            if success:
                print(f"✓ Report generation completed successfully!")
                print(f"  Report saved to: {self.report_path}")
                return True
            else:
                print(f"✗ Report generation failed")
                return False
            
        except Exception as e:
            print(f"✗ Error during report generation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_pipeline(self):
        """Run the complete analysis pipeline"""
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("  AI NARRATIVE NEXUS - DYNAMIC TEXT ANALYSIS PLATFORM")
        print("  Airline Sentiment Analysis Pipeline")
        print("="*70)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Preprocessing
        if not self.run_preprocessing():
            print("\n⚠️  Pipeline stopped due to preprocessing error")
            return False
        
        # Step 2: Sentiment Analysis
        if not self.run_sentiment_analysis():
            print("\n⚠️  Pipeline stopped due to sentiment analysis error")
            return False
        
        # Step 3: Topic Modeling
        if not self.run_topic_modeling():
            print("\n⚠️  Pipeline stopped due to topic modeling error")
            return False
        
        # Step 4: Insights Generation
        if not self.run_insights_generation():
            print("\n⚠️  Pipeline stopped due to insights generation error")
            return False
        
        # Step 5: Report Generation
        if not self.run_report_generation():
            print("\n⚠️  Pipeline stopped due to report generation error")
            return False
        
        # Complete
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.print_header("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"\nGenerated files:")
        print(f"  • Cleaned data: {self.clean_data_path}")
        print(f"  • Sentiment results: {self.sentiment_data_path}")
        print(f"  • Topic results: {self.topic_data_path}")
        print(f"  • Insights summary: {self.insights_path}")
        print(f"  • PDF report: {self.report_path}")
        print(f"\n✓ All analysis complete! Check the reports/ directory for outputs.\n")
        
        return True


def main():
    """Main entry point"""
    # Get base directory (src directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize pipeline
    pipeline = TextAnalysisPipeline(base_dir)
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("You can now run the Streamlit dashboard with:")
        print("  cd dashboard")
        print("  streamlit run app.py")
    
    return success


if __name__ == "__main__":
    main()
