# Project Parliament Pulse - Implementation Guide

## 🏛️ Project Overview
A privacy-first desktop application for parliamentarians to analyze constituent email inboxes using local data processing and machine learning.

## 🎯 MVP Scope & Architecture

### Core Principles
- **Privacy First**: All data processing occurs locally
- **Secure Access**: OAuth 2.0 only, no password storage
- **Desktop Native**: Packaged as a standalone application
- **Real-time Analysis**: Live dashboard with filtering capabilities

### Technical Stack Decision

#### Backend
- **Python 3.9+** - Core language
- **FastAPI** - Local web server and API
- **SQLite** - Local database (via sqlite3)
- **Pandas** - Data processing and SQLite interface
- **BERTopic** - Advanced topic modeling
- **vaderSentiment** - Fast sentiment analysis
- **BeautifulSoup4** - Email HTML parsing
- **Google API Client** - Gmail OAuth integration

#### Frontend (Recommended Modern Stack)
Instead of plain HTML/CSS/JS, we'll use:
- **Vite** - Modern build tool and dev server
- **TypeScript** - Type safety and better development experience
- **Tailwind CSS** - Utility-first CSS framework for clean, consistent design
- **Chart.js** - Robust charting library
- **Heroicons** - Beautiful SVG icons

This stack provides:
✅ Modern, clean design out of the box
✅ Excellent TypeScript support
✅ Fast development and build times
✅ Well-maintained and stable
✅ Great documentation and community

#### Desktop Packaging
- **Tauri** - Rust-based desktop app framework (more secure and lighter than Electron)

## 📋 Detailed Implementation Steps

### Phase 1: Project Foundation (Day 1)

#### Step 1.1: Directory Structure Setup
```
MP-Project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── database.py          # SQLite connection & models
│   │   ├── email_connector.py   # OAuth & email fetching
│   │   ├── nlp_processor.py     # NLP pipeline
│   │   └── config.py           # Configuration management
│   ├── requirements.txt
│   ├── .env.example
│   └── run_server.py           # Development server script
├── frontend/
│   ├── src/
│   │   ├── main.ts             # Application entry point
│   │   ├── style.css           # Global styles
│   │   ├── components/         # Reusable components
│   │   └── utils/              # Utility functions
│   ├── public/
│   │   └── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
├── src-tauri/                  # Tauri configuration (Phase 7)
├── data/                       # Local data storage
│   └── emails.db              # SQLite database
├── .gitignore
├── README.md
└── IMPLEMENTATION_GUIDE.md
```

#### Step 1.2: Backend Environment Setup
```bash
# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 1.3: Frontend Environment Setup
```bash
cd frontend
npm install
npm run dev  # Start development server
```

### Phase 2: Database Layer (Day 1-2)

#### Step 2.1: Database Schema Design
```sql
-- emails table
CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id TEXT UNIQUE NOT NULL,           -- Gmail message ID
    thread_id TEXT,                          -- Gmail thread ID
    sender_email TEXT NOT NULL,
    sender_name TEXT,
    subject TEXT NOT NULL,
    received_at DATETIME NOT NULL,
    raw_body TEXT NOT NULL,                  -- Original email body
    cleaned_body TEXT NOT NULL,              -- Processed email body
    is_spam BOOLEAN DEFAULT FALSE,
    topic TEXT,                              -- Assigned topic
    topic_confidence REAL,                   -- Topic confidence score
    sentiment_label TEXT,                    -- positive/neutral/negative
    sentiment_score REAL,                    -- Numerical sentiment score
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- topics table (for topic management)
CREATE TABLE topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    color TEXT,                              -- Hex color for UI
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- processing_status table (for tracking sync status)
CREATE TABLE processing_status (
    id INTEGER PRIMARY KEY,
    last_sync_at DATETIME,
    emails_processed INTEGER DEFAULT 0,
    last_email_date DATETIME
);
```

#### Step 2.2: Database Connection Manager
- Implement connection pooling
- Add migration system for schema updates
- Create data access layer with Pandas integration

### Phase 3: Email Integration (Day 2-3)

#### Step 3.1: Gmail OAuth Setup
```python
# Required OAuth scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# OAuth flow steps:
# 1. Generate authorization URL
# 2. Handle callback with authorization code
# 3. Exchange code for access token
# 4. Store refresh token securely (encrypted local file)
```

#### Step 3.2: Email Fetching Strategy
```python
# Batch processing approach:
# 1. Fetch emails in chunks (100-500 at a time)
# 2. Process incrementally to avoid API rate limits
# 3. Store raw data first, then process NLP
# 4. Implement resume capability for interrupted syncs
```

#### Step 3.3: Email Cleaning Pipeline
```python
# Multi-stage cleaning process:
# 1. HTML parsing and text extraction
# 2. Quote chain removal (regex patterns)
# 3. Signature detection and removal
# 4. Normalization (whitespace, encoding)
# 5. Language detection (English filter)
```

### Phase 4: NLP Processing Pipeline (Day 3-4)

#### Step 4.1: Spam Detection
```python
# Lightweight spam classifier:
# - Keyword-based filtering
# - Sender reputation checking
# - Subject line analysis
# - Body content patterns
```

#### Step 4.2: Topic Modeling with BERTopic
```python
# BERTopic configuration:
# - Use sentence-transformers for embeddings
# - UMAP for dimensionality reduction
# - HDBSCAN for clustering
# - Dynamic topic number detection
# - Minimum topic size: 10 emails
```

#### Step 4.3: Sentiment Analysis
```python
# vaderSentiment integration:
# - Handles social media text well
# - Fast processing for real-time analysis
# - Combines lexicon and rules-based approach
# - Returns compound score (-1 to +1)
```

### Phase 5: API Development (Day 4-5)

#### Step 5.1: Core API Endpoints
```python
# Authentication endpoints
POST /api/auth/gmail/start     # Initiate OAuth flow
GET  /api/auth/gmail/callback  # Handle OAuth callback
GET  /api/auth/status          # Check auth status

# Email processing endpoints
POST /api/emails/sync          # Trigger email sync
GET  /api/emails/status        # Sync progress status

# Dashboard data endpoints
GET  /api/dashboard/summary    # High-level KPIs
GET  /api/dashboard/trends     # Email volume over time
GET  /api/dashboard/topics     # Topic breakdown
GET  /api/dashboard/sentiment  # Sentiment analysis
```

#### Step 5.2: Data Aggregation Logic
```python
# Query patterns for dashboard:
# 1. Time-based grouping (day/week/month)
# 2. Topic aggregation with counts
# 3. Sentiment distribution calculations
# 4. Filtering by date ranges
# 5. Caching for performance
```

### Phase 6: Frontend Dashboard (Day 5-6)

#### Step 6.1: Component Architecture
```typescript
// Core components:
interface DashboardProps {
  dateRange: DateRange;
  granularity: 'day' | 'week' | 'month';
}

// Components:
- FilterControls      // Date picker and granularity selector
- KPICards           // Total emails, avg sentiment, topics count
- EmailTrendChart    // Line chart with Chart.js
- TopicsBarChart     // Horizontal bar chart
- SentimentTopicsChart // Grouped bar chart
- LoadingSpinner     // Processing indicator
```

#### Step 6.2: State Management
```typescript
// Simple state management with TypeScript:
interface AppState {
  isLoading: boolean;
  error: string | null;
  dateRange: { start: Date; end: Date };
  granularity: Granularity;
  dashboardData: DashboardData | null;
  authStatus: AuthStatus;
}
```

#### Step 6.3: Chart Configuration
```typescript
// Chart.js setup with TypeScript:
// - Responsive design
// - Dark/light theme support
// - Interactive tooltips
// - Export capabilities
// - Accessibility features
```

### Phase 7: Desktop Packaging with Tauri (Day 6-7)

#### Step 7.1: Tauri Setup
```bash
# Initialize Tauri
npm install --save-dev @tauri-apps/cli
npx tauri init

# Configure tauri.conf.json:
# - Set up Python backend as sidecar
# - Configure window properties
# - Set up auto-updater
# - Configure security policies
```

#### Step 7.2: Backend Integration
```rust
// Tauri commands for Python integration:
#[tauri::command]
async fn start_backend() -> Result<(), String> {
    // Start FastAPI server as background process
}

#[tauri::command]
async fn stop_backend() -> Result<(), String> {
    // Gracefully shutdown FastAPI server
}
```

#### Step 7.3: Build Configuration
```json
// tauri.conf.json configuration:
{
  "build": {
    "beforeBuildCommand": "npm run build",
    "beforeDevCommand": "npm run dev",
    "devPath": "http://localhost:3000",
    "distDir": "../frontend/dist"
  },
  "bundle": {
    "targets": ["app", "dmg", "msi"],
    "icon": ["icons/icon.png"]
  }
}
```

## 🚀 Development Workflow

### Daily Development Process
1. **Morning**: Plan day's objectives, review previous progress
2. **Code**: Implement features with test-driven approach
3. **Test**: Manual testing of new features
4. **Evening**: Commit progress, update documentation

### Testing Strategy
- **Unit Tests**: Core NLP functions and data processing
- **Integration Tests**: API endpoints and database operations
- **Manual Testing**: UI interactions and email processing
- **Performance Testing**: Large email dataset processing

### Security Considerations
- **OAuth Tokens**: Encrypted storage using system keyring
- **Local Database**: File permissions and encryption at rest
- **API Security**: CORS configuration and rate limiting
- **Desktop App**: Code signing and secure update mechanism

## 📊 Success Metrics for MVP

### Functional Requirements
✅ Successfully authenticate with Gmail OAuth
✅ Import and process 1000+ emails without errors
✅ Generate accurate topic classifications
✅ Display interactive dashboard with all charts
✅ Package as standalone desktop application

### Performance Requirements
✅ Email sync: Process 100 emails per minute
✅ Dashboard load: Display data within 2 seconds
✅ Topic modeling: Complete within 30 seconds for 1000 emails
✅ Memory usage: Under 500MB for typical usage

### User Experience Requirements
✅ Clean, intuitive interface design
✅ Responsive interactions (no UI freezing)
✅ Clear error messages and progress indicators
✅ Offline capability after initial sync

## 🔄 Future Enhancements (Post-MVP)

1. **Advanced NLP Features**
   - Named Entity Recognition (NER)
   - Intent classification
   - Multi-language support

2. **Enhanced Analytics**
   - Constituency mapping
   - Trend predictions
   - Comparative analysis

3. **Integration Expansion**
   - Microsoft 365 support
   - Social media monitoring
   - Calendar integration

4. **Collaboration Features**
   - Team dashboards
   - Report generation
   - Data export capabilities

This implementation guide provides a comprehensive roadmap for building the MVP systematically while maintaining code quality and user experience standards. 