# Text Summarization Feature

## Overview
The AI Narrative Nexus now includes advanced text summarization capabilities with both **Extractive** and **Abstractive** summarization methods.

## Features

### 1. Extractive Summarization
- Selects the most important sentences from the original text
- Uses TF-IDF scoring to rank sentence importance
- Configurable number of sentences (1-10)
- Preserves original phrasing and accuracy
- Best for: Maintaining exact wording and factual accuracy

### 2. Abstractive Summarization  
- Generates new condensed text using key phrases
- Word frequency-based importance scoring
- Configurable maximum words (20-200)
- Creates compressed versions of content
- Best for: Concise overviews and brevity

## How to Use

### Via Web Interface

1. **Navigate** to the "Summaries" page in the sidebar
2. **Choose** summary type (Extractive or Abstractive)
3. **Configure** parameters:
   - Extractive: Number of sentences per summary
   - Abstractive: Maximum words in summary
4. **Click** "Generate Summary" button
5. **View** results:
   - Overall summaries by sentiment category
   - Sample individual summaries

### Via API

#### Extractive Summary
```bash
POST http://localhost:5000/api/summary/extractive
Content-Type: application/json

{
  "num_sentences": 3
}
```

#### Abstractive Summary
```bash
POST http://localhost:5000/api/summary/abstractive
Content-Type: application/json

{
  "max_words": 50
}
```

## API Response Format

```json
{
  "success": true,
  "overall_summaries": {
    "positive": "Summary of positive sentiment texts...",
    "neutral": "Summary of neutral sentiment texts...",
    "negative": "Summary of negative sentiment texts..."
  },
  "total_summaries": 14640,
  "sample_summaries": [
    {
      "extractive_summary": "Example summary text..."
    }
  ]
}
```

## Technical Implementation

### Algorithm Details

**Extractive Summarization:**
- Sentence tokenization using NLTK
- TF-IDF vectorization for importance scoring
- Top-N sentence selection based on scores
- Maintains original sentence order

**Abstractive Summarization:**
- Word frequency analysis
- Stop word filtering
- Key phrase extraction
- Sentence compression and reconstruction

### Files Modified/Created

1. **Backend:**
   - `src/text_summarization.py` - Core summarization logic
   - `backend/app.py` - New API endpoints

2. **Frontend:**
   - `frontend/src/pages/SummaryPage.jsx` - UI component
   - `frontend/src/App.jsx` - Route registration

## Use Cases

- **Customer Feedback Analysis:** Quickly understand main points from reviews
- **Sentiment Reports:** Generate concise summaries by sentiment category
- **Quick Insights:** Get overview without reading full texts
- **Report Generation:** Include summaries in automated reports

## Requirements

- NLTK for sentence tokenization
- scikit-learn for TF-IDF vectorization
- Pandas for data processing

## Future Enhancements

- Integration with transformer models (BART, T5)
- Multi-document summarization
- Customizable summary styles
- Export summaries to various formats
- Real-time streaming summarization
