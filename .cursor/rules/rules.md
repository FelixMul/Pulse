# Cursor Rules for Project: Parliament Pulse

This document contains the standing rules and architectural decisions for our project. Please adhere to these guidelines in all your responses and code generation.

## 1. Project Mission
**Goal:** Build a secure, local-first desktop application for Parliamentarians to analyze their constituent email inbox using NLP.
**Core Principle:** **Privacy and Security.** All user data (emails, analysis results) MUST remain on the user's local machine. Do not suggest any cloud-based storage, processing, or external APIs for user data.

---

## 2. Core Architecture: "Local-First" Hybrid

This is the most important rule. Our application is a desktop app that runs a local web server internally.

1.  **Backend:** A **Python** application using the **FastAPI** framework. It will run a web server **only on `localhost`**. It is NOT a public web application.
2.  **Database:** **SQLite**. We will use Python's built-in `sqlite3` library, often managed via the **Pandas** library (`df.to_sql`, `pd.read_sql_query`).
    -   **Explicitly DO NOT use PostgreSQL, MySQL, or any other server-based database.**
    -   The database will be a single file (e.g., `parliament_pulse.db`) stored on the user's computer.
3.  **Frontend:** A single-page web interface built with plain **HTML, CSS, and JavaScript**. It will run inside a webview.
4.  **Desktop Wrapper:** **Tauri**. We will use Tauri to bundle the Python backend and the web frontend into a single, executable desktop application for macOS and Windows.

---

## 3. Technical Stack

Adhere strictly to this stack.

-   **Backend Language:** Python 3.9+
-   **Backend Framework:** FastAPI
-   **Data Manipulation:** Pandas
-   **Database:** SQLite (via Python's `sqlite3`)
-   **NLP Libraries:**
    -   **Topic Modeling:** `bertopic`
    -   **Sentiment Analysis:** `vaderSentiment` (for the MVP)
    -   **Email Parsing:** `beautifulsoup4` (for HTML), `regex` (for signatures)
-   **Email API Clients:** `google-api-python-client`, `google-auth-oauthlib`
-   **Frontend:** Vanilla JavaScript (ES6+), HTML5, CSS3
-   **Charting Library:** Chart.js
-   **Packaging:** Tauri

---

## 4. File Structure & Naming Conventions

-   **Root Directories:**
    -   `backend/`: All Python source code.
    -   `frontend/`: All HTML, CSS, and JS files.
    -   `src-tauri/`: Tauri configuration files (we will set this up later).
-   **Python Naming:** `snake_case` for files and variables (e.g., `nlp_processor.py`). Use modular files as planned (`main.py`, `database.py`, `nlp_processor.py`, `email_connector.py`).
-   **JavaScript Naming:** `camelCase` for variables and functions.
-   **Security:** Never hardcode secrets (API keys, client secrets). Use a `.env` file and the `python-dotenv` library for local development.

---

## 5. MVP Feature Scope

Focus only on the following MVP features. Do not suggest or implement features from the "Future Ideas" list unless specifically asked.

-   **Analysis:** Topic Modeling and Sentiment Analysis are the core.
-   **Dashboard:**
    -   Date Range & Granularity Filters.
    -   KPIs: Total Emails, Avg Sentiment.
    -   Charts: Email Volume (Line), Top Topics (Bar), Sentiment per Topic (Grouped Bar).
-   **Data Source:** OAuth 2.0 connection to Gmail/Microsoft 365.

---

## 6. Project Roadmap (Our Current Plan)

Refer to this plan to understand our current focus. We will proceed step-by-step.

-   [ ] **Step 1: Project Setup & Backend Foundation:** Initialize FastAPI, directory structure, and dependencies.
-   [ ] **Step 2: Database and Data Models:** Set up the SQLite connection and schema.
-   [ ] **Step 3: Email Ingestion & Cleaning:** Implement the OAuth flow and email parsing logic.
-   [ ] **Step 4: The Core NLP Pipeline:** Create functions for topic and sentiment analysis and integrate them.
-   [ ] **Step 5: API Endpoints:** Build the FastAPI endpoint to serve aggregated dashboard data.
-   [ ] **Step 6: Frontend Dashboard:** Develop the HTML/CSS/JS interface and connect it to the API.
-   [ ] **Step 7: Packaging with Tauri:** Bundle everything into a final desktop application.

We are currently at **Step 1**.