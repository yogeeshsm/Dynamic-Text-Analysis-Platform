# SQLite Database Implementation

## ✅ **Successfully Implemented!**

Your platform now uses **SQLite database** to store all processed data, analysis results, and session history.

---

## 📁 Database Location

- **File**: `data/analysis.db`
- **Full Path**: `C:\Users\S M Yogesh\OneDrive\ドキュメント\dynamic text analysis platform\AI_Narrative_Nexus\data\analysis.db`

---

## 🗄️ Database Schema

### 1. **sessions** table
Stores analysis session metadata
```sql
- id (INTEGER PRIMARY KEY)
- session_name (TEXT)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
- status (TEXT) - 'active', 'completed', 'error'
- total_records (INTEGER)
- dataset_info (JSON)
```

### 2. **raw_data** table
Stores uploaded raw text data
```sql
- id (INTEGER PRIMARY KEY)
- session_id (INTEGER FK -> sessions)
- text (TEXT) - original text
- metadata (JSON) - airline, date, etc.
- created_at (TIMESTAMP)
```

### 3. **preprocessed_data** table
Stores cleaned and preprocessed text
```sql
- id (INTEGER PRIMARY KEY)
- session_id (INTEGER FK -> sessions)
- raw_data_id (INTEGER FK -> raw_data)
- original_text (TEXT)
- cleaned_text (TEXT)
- tokens (JSON) - list of tokens
- created_at (TIMESTAMP)
```

### 4. **sentiment_results** table
Stores sentiment analysis results
```sql
- id (INTEGER PRIMARY KEY)
- session_id (INTEGER FK -> sessions)
- preprocessed_id (INTEGER FK -> preprocessed_data)
- text (TEXT)
- sentiment_label (TEXT) - 'positive', 'negative', 'neutral'
- sentiment_score (REAL) - -1.0 to 1.0
- confidence (REAL) - 0.0 to 1.0
- method (TEXT) - 'vader', 'textblob', 'distilbert'
- created_at (TIMESTAMP)
```

### 5. **topic_results** table
Stores topic modeling results
```sql
- id (INTEGER PRIMARY KEY)
- session_id (INTEGER FK -> sessions)
- sentiment_id (INTEGER FK -> sentiment_results)
- text (TEXT)
- topic_id (INTEGER)
- topic_label (TEXT) - e.g., 'Customer Service Issues'
- topic_probability (REAL)
- method (TEXT) - 'lda', 'nmf'
- created_at (TIMESTAMP)
```

### 6. **insights** table
Stores generated insights
```sql
- id (INTEGER PRIMARY KEY)
- session_id (INTEGER FK -> sessions)
- insight_type (TEXT) - 'airline_ranking', 'keyword', 'summary'
- insight_data (JSON) - structured insight data
- created_at (TIMESTAMP)
```

---

## 🔄 **How It Works**

### Workflow:
1. **Upload Data** → Saved to `raw_data` table with new `session_id`
2. **Preprocessing** → Cleaned text saved to `preprocessed_data` table
3. **Sentiment Analysis** → Results saved to `sentiment_results` table
4. **Topic Modeling** → Results saved to `topic_results` table
5. **Insights** → Generated insights saved to `insights` table

### All steps are linked by `session_id` for easy retrieval!

---

## 🔌 **New API Endpoints**

### Session Management

1. **List All Sessions**
   ```http
   GET /api/sessions/list
   ```
   Returns: All analysis sessions with metadata

2. **Get Session Details**
   ```http
   GET /api/sessions/<session_id>
   ```
   Returns: Complete session data (raw, preprocessed, sentiment, topics, insights)

3. **Delete Session**
   ```http
   DELETE /api/sessions/<session_id>
   ```
   Returns: Success confirmation

---

## 💡 **Benefits**

✅ **Persistent Storage** - All data saved permanently
✅ **Session History** - Track multiple analysis runs
✅ **Fast Queries** - SQLite is optimized for retrieval
✅ **Data Integrity** - Foreign key constraints ensure consistency
✅ **Easy Export** - Query and export any session data
✅ **Analysis Comparison** - Compare results across sessions
✅ **No Data Loss** - Even if server restarts, data is preserved

---

## 🚀 **Current Status**

✅ Database schema created
✅ Backend integrated with database
✅ All endpoints updated to use SQLite
✅ Session management implemented
✅ Backend server running on port 5000
✅ Frontend server running on port 3000

---

## 📊 **Access Your Application**

- **Frontend**: http://localhost:3000
- **Backend API**: http://127.0.0.1:5000
- **Database**: Use any SQLite browser to view `data/analysis.db`

---

## 🔧 **Database Tools** (Optional)

You can view and query the database using:
- **DB Browser for SQLite** (Free GUI tool)
- **VSCode SQLite Extension**
- **Python**: `sqlite3` module
- **Command Line**: `sqlite3 data/analysis.db`

---

## 📝 **Example Queries**

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('data/analysis.db')
cursor = conn.cursor()

# Get all sessions
cursor.execute("SELECT * FROM sessions")
sessions = cursor.fetchall()

# Get sentiment results for session 1
cursor.execute("""
    SELECT text, sentiment_label, sentiment_score 
    FROM sentiment_results 
    WHERE session_id = 1
""")
results = cursor.fetchall()

conn.close()
```

---

## 🎯 **Next Steps**

The platform is now ready to use with full database integration! All your analysis data will be:
- ✅ Automatically saved to SQLite
- ✅ Retrievable anytime
- ✅ Organized by sessions
- ✅ Queryable through API or SQL

**Enjoy your enhanced platform!** 🚀
