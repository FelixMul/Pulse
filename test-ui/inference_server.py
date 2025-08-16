#!/usr/bin/env python3
"""
Simple inference server for Parliament Pulse model testing.
Loads the trained DistilBERT topic model and provides VADER sentiment analysis.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add the Model Finetuning directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ModelFinetuning', 'finetuning')))

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="Parliament Pulse Inference Server", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
topic_model = None
topic_tokenizer = None
topic_names = None
sentiment_analyzer = None

# Request/Response models
class PredictionRequest(BaseModel):
    email_text: str

class TopicPrediction(BaseModel):
    topic: str
    confidence: float

class PredictionResponse(BaseModel):
    topics: List[TopicPrediction]
    sentiment: Dict[str, Any]
    success: bool
    error: str = None

@app.on_event("startup")
async def load_models():
    """Load models on server startup."""
    global topic_model, topic_tokenizer, topic_names, sentiment_analyzer
    
    try:
        print("🔄 Loading models...")
        
        # Load topic classification model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ModelFinetuning', 'finetuning', 'models', 'topic_model'))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        print(f"📂 Loading topic model from: {model_path}")
        topic_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        topic_model = DistilBertForSequenceClassification.from_pretrained(model_path)
        topic_model.eval()
        
        # Load topic names (from the training process)
        # Since we don't have them saved, we'll use the standard UK parliamentary topics
        topic_names = [
            'Agriculture & Rural Affairs', 'Animal Welfare', 'Business & Enterprise',
            'Childcare & Family Support', 'Consumer Rights & Issues', 'Cost of Living & Economy',
            'Crime & Community Safety', 'Culture, Media & Sport', 'Defence & National Security',
            'Digital & Technology', 'Disability Rights & Access', 'Education & Schools',
            "Employment & Workers' Rights", 'Energy & Utilities', 'Environment & Climate Change',
            'Foreign Affairs & International Development', 'Healthcare & NHS', 'Housing & Planning',
            'Immigration & Asylum', 'Justice & Legal System', 'Local Campaign Support',
            'Local Government & Council Tax', 'Mental Health Services', 'Non-Actionable / Incoherent',
            'Pensions & National Insurance', 'Planning & Development', 'Social Security & Benefits',
            'Taxation & Public Spending', 'Trade & Brexit Issues', 'Transportation & Infrastructure'
        ]
        
        print(f"✅ Topic model loaded with {len(topic_names)} topics")
        
        # Load VADER sentiment analyzer
        print("📊 Loading VADER sentiment analyzer...")
        sentiment_analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER sentiment analyzer loaded")
        
        # Check device
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        topic_model.to(device)
        print(f"🖥️  Using device: {device}")
        
        print("🚀 All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Parliament Pulse Inference Server",
        "models_loaded": topic_model is not None and sentiment_analyzer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_email(request: PredictionRequest):
    """
    Predict topics and sentiment for an email.
    
    Args:
        request: Email text to analyze
        
    Returns:
        Topics with confidence scores and sentiment analysis
    """
    try:
        if not topic_model or not sentiment_analyzer:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        email_text = request.email_text.strip()
        if not email_text:
            raise HTTPException(status_code=400, detail="Email text cannot be empty")
        
        # Predict topics
        topic_predictions = predict_topics(email_text)
        
        # Predict sentiment using VADER
        sentiment_result = predict_sentiment(email_text)
        
        return PredictionResponse(
            topics=topic_predictions,
            sentiment=sentiment_result,
            success=True
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return PredictionResponse(
            topics=[],
            sentiment={},
            success=False,
            error=str(e)
        )

def predict_topics(email_text: str, threshold: float = 0.1) -> List[TopicPrediction]:
    """
    Predict topics for email text using the trained model.
    
    Args:
        email_text: The email content
        threshold: Confidence threshold for topic inclusion
        
    Returns:
        List of topic predictions with confidence scores
    """
    try:
        # Tokenize input
        inputs = topic_tokenizer(
            email_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(topic_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = topic_model(**inputs)
            logits = outputs.logits
            
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create topic predictions above threshold
        predictions = []
        for i, (topic_name, confidence) in enumerate(zip(topic_names, probabilities)):
            if confidence >= threshold:
                predictions.append(TopicPrediction(
                    topic=topic_name,
                    confidence=float(confidence)
                ))
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        # If no topics above threshold, return top 5 with lower confidence
        if not predictions:
            top_indices = np.argsort(probabilities)[-5:][::-1]
            predictions = [
                TopicPrediction(
                    topic=topic_names[i],
                    confidence=float(probabilities[i])
                )
                for i in top_indices if probabilities[i] > 0.01  # Very low threshold for fallback
            ]
        
        return predictions
        
    except Exception as e:
        print(f"❌ Topic prediction error: {e}")
        return []

def predict_sentiment(email_text: str) -> Dict[str, Any]:
    """
    Predict sentiment using VADER.
    
    Args:
        email_text: The email content
        
    Returns:
        Dictionary with sentiment scores and classification
    """
    try:
        # Get VADER scores
        scores = sentiment_analyzer.polarity_scores(email_text)
        
        # Debug print to understand the 0.382 issue
        print(f"🔍 VADER scores for text length {len(email_text)}: {scores}")
        
        # Determine overall sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Calculate confidence differently - use max of pos/neg/neu as base confidence
        max_component = max(scores['pos'], scores['neg'], scores['neu'])
        confidence = min(max_component + abs(compound) * 0.5, 1.0)  # Blend component and compound
        
        return {
            "sentiment": sentiment,
            "compound": compound,
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu'],
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"❌ Sentiment prediction error: {e}")
        return {
            "sentiment": "Unknown",
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "confidence": 0.0
        }

@app.get("/topics")
async def get_available_topics():
    """Get list of available topic categories."""
    return {
        "topics": topic_names,
        "count": len(topic_names) if topic_names else 0
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "topic_model_loaded": topic_model is not None,
        "sentiment_analyzer_loaded": sentiment_analyzer is not None,
        "available_topics": len(topic_names) if topic_names else 0,
        "device": str(next(topic_model.parameters()).device) if topic_model else "none"
    }

if __name__ == "__main__":
    print("🏛️ Starting Parliament Pulse Inference Server...")
    print("📍 Access at: http://localhost:8001")
    print("📊 API docs at: http://localhost:8001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info") 