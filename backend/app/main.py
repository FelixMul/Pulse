"""
Parliament Pulse - Beta Version
Simple sentiment analysis API for testing
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

from .config import settings
from .database import db_manager
from .nlp_processor import NLPProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global NLP processor instance (loaded at startup)
nlp_processor = None

# Request/Response models
class AnalyzeRequest(BaseModel):
    text: str
    sender_email: Optional[str] = "test@example.com"
    subject: Optional[str] = "Test Email"

class AnalyzeResponse(BaseModel):
    success: bool
    analysis: dict
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="Parliament Pulse - Beta",
    description="Sentiment analysis for email content",
    version="0.1-beta",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS - permissive for beta testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for beta testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize NLP processor and models at startup"""
    global nlp_processor
    logger.info("Loading NLP processor and models at startup...")
    try:
        nlp_processor = NLPProcessor()
        logger.info("✅ NLP processor initialized successfully")
        
        # Log model status
        status = nlp_processor.get_model_status()
        for model_name, model_status in status.items():
            ready_status = "✅ Ready" if model_status.get('ready', False) else "❌ Not Ready"
            logger.info(f"  {model_name}: {ready_status}")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize NLP processor: {str(e)}")
        # Don't crash the app, but make sure to handle this in the endpoints

@app.get("/")
async def root():
    """Root endpoint with beta information"""
    return JSONResponse(
        status_code=200,
        content={
            "name": "Parliament Pulse - Beta",
            "version": "0.1-beta",
            "description": "Test the sentiment analysis functionality",
            "status": "ready for testing",
            "endpoints": {
                "analyze": "/api/analyze - POST text for sentiment analysis",
                "health": "/api/health - System health check",
                "docs": "/docs - API documentation"
            }
        }
    )

@app.get("/api/health")
async def health_check():
    """Simple health check"""
    try:
        # Test NLP processor
        if nlp_processor is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not ready",
                    "nlp_ready": False,
                    "database": "connected",
                    "version": "0.1-beta",
                    "message": "NLP processor not initialized"
                }
            )
            
        nlp_status = nlp_processor.get_model_status()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "nlp_ready": nlp_status['sentiment_analyzer']['ready'],
                "database": "connected",
                "version": "0.1-beta"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text for sentiment, spam detection, and topic classification
    This is the core functionality for beta testing
    """
    try:
        # Check if NLP processor is ready
        if nlp_processor is None:
            raise HTTPException(
                status_code=503,
                detail="NLP processor not initialized. Please wait for startup to complete."
            )
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Process the text with our NLP pipeline
        logger.info(f"Processing text analysis for ID: {analysis_id}")
        
        # Run the full NLP analysis
        results = nlp_processor.process_email(
            email_text=request.text,
            email_id=analysis_id
        )
        
        # Structure the response for easy frontend consumption
        analysis_data = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "text_length": len(request.text),
                "sender": request.sender_email,
                "subject": request.subject
            },
            "sentiment": results.get("sentiment_analysis", {}),
            "spam_detection": results.get("spam_detection", {}),
            "topic_modeling": results.get("topic_modeling", {}),
            "text_processing": results.get("text_features", {})
        }
        
        # Optionally save to database for testing history
        try:
            email_data = {
                'id': analysis_id,
                'email_id': analysis_id,
                'sender_email': request.sender_email,
                'sender_name': None,
                'subject': request.subject,
                'received_at': datetime.now().isoformat(),
                'raw_body': request.text,
                'cleaned_body': results.get("cleaned_text", ""),
                'topic': results.get("topic_modeling", {}).get("topic_label", ""),
                'topic_id': results.get("topic_modeling", {}).get("topic_id", -1),
                'sentiment': results.get("sentiment_analysis", {}).get("sentiment", ""),
                'sentiment_compound': results.get("sentiment_analysis", {}).get("scores", {}).get("compound", 0.0),
                'sentiment_positive': results.get("sentiment_analysis", {}).get("scores", {}).get("positive", 0.0),
                'sentiment_negative': results.get("sentiment_analysis", {}).get("scores", {}).get("negative", 0.0),
                'sentiment_neutral': results.get("sentiment_analysis", {}).get("scores", {}).get("neutral", 0.0),
                'is_spam': results.get("spam_detection", {}).get("is_spam", False),
                'spam_confidence': results.get("spam_detection", {}).get("confidence", 0.0),
                'processed_at': datetime.now().isoformat()
            }
            db_manager.insert_email(email_data)
            logger.info(f"Saved analysis results to database: {analysis_id}")
        except Exception as db_error:
            logger.warning(f"Could not save to database: {str(db_error)}")
        
        return AnalyzeResponse(
            success=True,
            analysis=analysis_data,
            message="Analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/recent-analyses")
async def get_recent_analyses(limit: int = 10):
    """Get recent analysis results for testing"""
    try:
        recent = db_manager.get_recent_emails(limit=limit)
        return JSONResponse(
            status_code=200,
            content={
                "recent_analyses": recent,
                "count": len(recent)
            }
        )
    except Exception as e:
        logger.error(f"Failed to get recent analyses: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, debug=settings.DEBUG) 