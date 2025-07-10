# Parliament Pulse - Test UI

A simple testing interface for the Parliament Pulse email analysis toolkit.

## How to Use

### 1. Start the Backend Server

First, make sure the backend server is running:

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Open the Test Interface

Simply open `index.html` in your web browser:

```bash
open test-ui/index.html
```

Or double-click the `index.html` file to open it in your default browser.

### 3. Test Email Analysis

1. **Enter Email Content**: Paste any email text into the large text area
2. **Optional Fields**: Add subject line and sender email if desired
3. **Click "Analyze Email"**: The system will process the text and show results

## Features Tested

The test UI shows results for all core analysis features:

- **😊 Sentiment Analysis**: Positive/Negative/Neutral with confidence scores
- **🛡️ Spam Detection**: Spam detection with confidence percentage
- **🏷️ Topic Classification**: Topic identification and keywords
- **🔧 Text Processing**: Word count, reading level, cleaned text preview

## Sample Test Cases

Try these examples to see different analysis results:

### Positive Email
```
Dear [MP Name],
I want to thank you for your excellent work on the new education bill. The proposed funding increases for our local schools will make a tremendous difference for our children's future. Keep up the great work!
```

### Negative Email
```
I am extremely disappointed with the recent healthcare policy changes. The cuts to rural medical services will devastate our community. This is completely unacceptable and I demand immediate action.
```

### Potential Spam
```
URGENT!!! You have won $1,000,000!!! Click here now to claim your prize!!! Limited time offer!!! Act fast!!!
```

## Troubleshooting

- **"Failed to analyze email"**: Make sure the backend server is running on `http://127.0.0.1:8000`
- **Empty results**: Check browser console for error messages
- **Slow analysis**: First analysis may be slower as ML models load into memory

## Purpose

This test UI is separate from the main frontend dashboard to allow:
- Quick testing of analysis functionality
- Beta testing without complexity
- Standalone demonstration of core features
- Development iteration without affecting main UI 