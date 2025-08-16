#  Pulse

**Privacy-first email analysis for parliamentarians**

A desktop application that helps parliamentarians analyze their constituent email inbox using local data processing and machine learning. All data remains on your local machine for maximum privacy and security.

## Features

- **Secure Email Integration**: OAuth 2.0 authentication with Gmail (no password storage)
- **Privacy-First**: All processing happens locally on your machine
- **NLP Analysis**: Topic modeling, sentiment analysis, and spam filtering
- **Interactive Dashboard**: Real-time analytics with beautiful charts
- **Desktop App**: Standalone application for macOS and Windows

## Quick Start

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

### 3. Setup Local LLM (Optional for Step 1)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull gpt-oss 20B model (will be used in Step 2)
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

## Dashboard Features

### High-Level KPIs
- Total email count for selected period
- Average sentiment score
- Number of unique topics identified

### Interactive Charts
- **Email Volume Trend**: Line chart showing email volume over time
- **Top Topics**: Bar chart of most discussed topics
- **Sentiment by Topic**: Grouped bar chart showing sentiment distribution per topic

### Filtering Options
- Date range picker
- Time granularity (day/week/month)
- Real-time data refresh

## Development

### Project Structure

```
MP-Project/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── database.py     # SQLite database management
│   │   ├── config.py       # Configuration settings
│   │   ├── email_connector.py    # Gmail OAuth integration
│   │   └── nlp_processor.py      # NLP processing pipeline
│   └── requirements.txt    # Python dependencies
├── frontend/               # Vite + TypeScript frontend
│   ├── src/
│   │   ├── main.ts        # Application entry point
│   │   └── style.css      # Tailwind CSS styles
│   ├── public/
│   │   └── index.html     # Main HTML file
│   └── package.json       # Node.js dependencies
├── data/                  # Local SQLite database
└── IMPLEMENTATION_GUIDE.md # Detailed development guide
```

### Technology Stack

**Backend:**
- FastAPI (Python web framework)
- SQLite (local database)
- Fine-tuned DistilBERT (custom topic classification)
- custom sentiment analysis -> not yet decided
- HuggingFace Transformers (model framework)
- Google API Client (Gmail integration)

**Frontend:**
- Vite (build tool)
- TypeScript (type safety)
- Tailwind CSS (styling)
- Chart.js (data visualization)

**Desktop Packaging:**
- Tauri (Rust-based app framework)

### Development Workflow

1. **Backend Development**: Start with `python backend/run_server.py`
2. **Frontend Development**: Start with `npm run dev` in frontend directory
3. **API Documentation**: Visit http://127.0.0.1:8000/docs when backend is running
4. **Type Checking**: Run `npm run type-check` in frontend directory

## 🔒 Privacy & Security

- **Local Processing**: All email data and NLP processing stays on your machine
- **OAuth 2.0**: Secure authentication without password storage
- **Encrypted Storage**: Tokens stored securely using system keyring
- **No External APIs**: No data sent to third-party services for analysis

## Current Status

✅ **Step 1**: Simplified development environment with unified startup  
✅ **Step 2**: Local LLM integration with gpt-oss 20B for email analysis  
🔄 **Step 3**: Generate synthetic time-series email dataset (Next)  
📋 **Step 4**: Add time-series visualization to frontend  
📋 **Step 5**: MCP integration (low priority)

###  Recent Progress: Custom ML Models

 **Synthetic Dataset**: Generated 1,250 labeled emails with topics, sentiment, and personas  
 **Model Training Pipeline**: DistilBERT fine-tuning framework for topic and sentiment classification  
 **Model Training**: In progress - training custom models on domain-specific data  
 **Model Integration**: Integrate trained models into backend NLP processor  

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed development roadmap.

## ⚠️ Model Limitation: 512-Token Input Truncation

**Important:** DistilBERT (and BERT) models have a maximum input length of 512 tokens. Our analysis shows that the average email in the dataset is about 500 tokens, meaning many emails are truncated during training and inference. This can result in loss of important information, especially for longer emails.

### Potential Problems
- The end of longer emails is cut off, possibly omitting key context or requests.
- Model performance may be reduced for emails where critical information is not in the first 512 tokens.

### Possible Solutions
- **Chunking:** Split long emails into multiple 512-token segments, run the model on each, and aggregate predictions (e.g., majority vote, max confidence).
- **Summarization:** Preprocess long emails with an automatic summarizer to fit within the token limit.
- **Prioritize Sections:** Use only the most relevant part of the email (e.g., introduction and conclusion) if domain knowledge allows.

> **Note:** The current implementation uses truncation, so only the first 512 tokens of each email are used by the model.

##  Contributing

This is a collaborative development project. Each phase builds upon the previous one with careful attention to:

- **Privacy by design**
- **Local-first architecture**
- **Type safety and error handling**
- **Clean, maintainable code**
- **Comprehensive testing**

## 📄 License

This project is private and proprietary. All rights reserved.

---

**Parliament Pulse** - Empowering parliamentarians with privacy-first email analytics

---

## Test UI: Quick Start Guide

To run the test UI and backend for model testing:

1. **Start the backend inference server:**
   ```bash
   python test-ui/inference_server.py
   ```
   This will start the backend at http://localhost:8080

2. **Start a static server for the frontend (from the project root):**
   ```bash
   python3 -m http.server 8080
   ```
   This will serve your files at http://localhost:8080

3. **Open the test UI in your browser:**
   [http://localhost:8080/test-ui/index.html](http://localhost:8080/test-ui/index.html)

- The UI will now be able to communicate with the backend for predictions and analysis.
- If you see any errors, ensure both servers are running and check your browser console for details.

