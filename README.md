# 🏛️ Parliament Pulse - Local Email Analysis POC

A proof-of-concept system for analyzing constituent emails using local LLM processing. Built for parliamentarians to understand constituent concerns without external data processing.

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+**
- **UV package manager** (fast pip replacement)
- **Ollama** (for local LLM)

### 1. Install UV
```bash
# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies
```bash
# Install project dependencies
uv sync
```

### 3. Setup Local LLM
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull gpt-oss 20B model
ollama pull gpt-oss:20b
```

### 4. Start the Application
```bash
# Single command to start everything
uv run python start_server.py
```

This will:
- Start the backend API on port 8080
- Serve the test UI at the same address
- Automatically open your browser to http://localhost:8080

**That's it!** The application will be running with a unified interface.

## 🎯 What This POC Does

### Core Features
- **Local Email Analysis**: Uses gpt-oss 20B via Ollama for topic extraction and sentiment analysis
- **No External APIs**: All processing happens on your machine
- **Comprehensive Classification**: 29 topic categories covering UK political issues
- **Sentiment Analysis**: 5-level scale (very negative to very positive)
- **Smart Caching**: Avoids re-analyzing the same emails
- **Test Interface**: Simple web UI for testing and demonstration

### Technical Stack
- **Backend**: FastAPI with local JSON storage
- **LLM**: gpt-oss 20B running locally via Ollama
- **Frontend**: Simple HTML/JS test interface
- **Storage**: Local JSON files (no database required)
- **Package Management**: UV for fast dependency management

## 📊 Current Capabilities

### Email Analysis
- **Topic Classification**: Automatically categorizes emails into 29 political themes
- **Sentiment Scoring**: Determines constituent satisfaction level
- **Confidence Metrics**: Shows how certain the AI is about its analysis
- **Summary Generation**: Creates concise summaries of email content

### Sample Topics
- Healthcare & NHS
- Housing & Planning  
- Immigration & Asylum
- Education & Schools
- Cost of Living & Economy
- Social Security & Benefits
- Transportation & Infrastructure
- Environment & Climate
- Local Campaign Support
- And 20 more categories...

### Sample Sentiments
- Very Negative
- Negative  
- Neutral
- Positive
- Very Positive

## 🔧 Development Status

### ✅ Completed
- **Step 1**: Unified development environment with single startup command
- **Step 2**: Local LLM integration with gpt-oss 20B
  - Fixed critical CSV parsing issues (was truncating emails)
  - Implemented robust JSON parsing for LLM output
  - Added comprehensive debugging and error handling
  - Built email caching system

### 🔄 Next Steps
- **Step 3**: Generate synthetic time-series email dataset
- **Step 4**: Add time-series visualization to frontend
- **Step 5**: MCP integration (low priority)

## 📁 Project Structure

```
MP-Project/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # API endpoints
│   │   ├── llm_processor.py # gpt-oss 20B integration
│   │   ├── simple_storage.py # Local JSON storage
│   │   └── config.py       # Configuration
├── test-ui/                 # Simple test interface
│   └── index.html          # Frontend for testing
├── data/                    # Email datasets and analysis results
├── start_server.py          # Unified startup script
└── pyproject.toml          # UV dependencies
```

## 🧪 Testing the System

1. **Start the server**: `uv run python start_server.py`
2. **Open browser**: Navigate to http://localhost:8080
3. **Load test emails**: Click "Load Test Emails" to load 250 sample emails
4. **Analyze emails**: Click on any email to run LLM analysis
5. **View results**: See topic, sentiment, confidence, and summary

## 🔍 Debugging

The system includes comprehensive debugging:
- **Terminal output**: Shows full LLM responses and processing steps
- **Email content**: Displays actual content being sent to LLM
- **Analysis pipeline**: Traces each step of the analysis process

## 📈 Performance

- **Email Processing**: Handles emails up to 4000 characters (was 1500)
- **Analysis Speed**: ~10-15 seconds per email with gpt-oss 20B
- **Cache Efficiency**: Avoids redundant analysis with email ID tracking
- **Success Rate**: Significantly improved with full email context

## 🚧 Limitations (POC Version)

- **Simple UI**: Basic test interface (will be improved in Step 4)
- **Local Only**: Requires Ollama and gpt-oss 20B locally
- **No Authentication**: Basic security (suitable for development)
- **Limited Dataset**: Currently 250 test emails (will expand in Step 3)

## 🤝 Contributing

This is a proof-of-concept project. The focus is on demonstrating local LLM capabilities for political email analysis.

## 📄 License

Project-specific license - see project documentation for details.

---

**Built for parliamentarians who need local, private email analysis without external dependencies.**

