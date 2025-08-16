# Parliament Pulse - Test UI

A minimal interface for testing the Parliament Pulse email analysis backend.

## Quick Start

### 1. Start the Backend Inference Server
From the project root, run:
```bash
python test-ui/inference_server.py
```
This starts the backend at: http://localhost:8001

### 2. Serve the Frontend via Static Server
From the project root, run:
```bash
python3 -m http.server 8080
```
This serves the UI at: http://localhost:8080/test-ui/index.html

### 3. Open the Test UI
In your browser, go to:
```
http://localhost:8080/test-ui/index.html
```

- The UI loads test emails from `/data/email_dataset_test.csv` and sends analysis requests to the backend.
- You can select emails to view model predictions and ground truth labels.

## Features
- **Topic Classification**: Predicts topics for each email
- **Sentiment Analysis**: VADER-based sentiment prediction
- **Accuracy Assessment**: Compares predictions to ground truth

## Troubleshooting
- If you see "Failed to fetch" or loading errors, ensure both servers are running and you are accessing the UI via `http://localhost:8080` (not as a file).
- The backend must be running on port 8001.
- Check the browser console and terminal for error messages.

## Purpose
This test UI is for rapid evaluation of the Parliament Pulse models and is separate from the main dashboard frontend. 