# 🏛️ Parliament Pulse

**Privacy-first email analysis for parliamentarians**

A desktop application that helps parliamentarians analyze their constituent email inbox using local data processing and machine learning. All data remains on your local machine for maximum privacy and security.

## 🎯 Features

- **Secure Email Integration**: OAuth 2.0 authentication with Gmail (no password storage)
- **Privacy-First**: All processing happens locally on your machine
- **NLP Analysis**: Topic modeling, sentiment analysis, and spam filtering
- **Interactive Dashboard**: Real-time analytics with beautiful charts
- **Desktop App**: Standalone application for macOS and Windows

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (for backend)
- **Node.js 18+** (for frontend)
- **Git** (for version control)

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
python run_server.py
```

The backend API will be available at: http://127.0.0.1:8000

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at: http://127.0.0.1:3000

### 3. Configuration

1. Copy `backend/.env.example` to `backend/.env`
2. Configure your Google OAuth credentials (see [OAuth Setup Guide](IMPLEMENTATION_GUIDE.md#step-31-gmail-oauth-setup))
3. Adjust other settings as needed

## 📊 Dashboard Features

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

## 🛠️ Development

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

## 📈 Current Status

✅ **Phase 1**: Project foundation and basic FastAPI backend  
✅ **Phase 2**: Database layer and email integration  
🚧 **Phase 3**: NLP processing pipeline with custom model training  
⏳ **Phase 4**: Dashboard API endpoints  
⏳ **Phase 5**: Frontend dashboard completion  
⏳ **Phase 6**: Tauri desktop packaging

### 🎯 Recent Progress: Custom ML Models

✅ **Synthetic Dataset**: Generated 1,250 labeled emails with topics, sentiment, and personas  
✅ **Model Training Pipeline**: DistilBERT fine-tuning framework for topic and sentiment classification  
🚧 **Model Training**: In progress - training custom models on domain-specific data  
⏳ **Model Integration**: Integrate trained models into backend NLP processor  

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed development roadmap.

## 🤝 Contributing

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

