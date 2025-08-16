"""
Parliament Pulse - POC Version
Local LLM-based email analysis API
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from .config import settings
from .simple_storage import storage
from .llm_processor import llm_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class AnalyzeRequest(BaseModel):
    text: str
    sender_email: Optional[str] = "test@example.com"
    subject: Optional[str] = "Test Email"
    email_id: Optional[str] = None  # Optional unique identifier for the email

class AnalyzeResponse(BaseModel):
    success: bool
    id: str
    topic: str
    sentiment: str
    confidence: float
    summary: str
    timestamp: str
    message: str

class StatsResponse(BaseModel):
    total_analyses: int
    topics: Dict[str, int]
    sentiments: Dict[str, int]
    average_confidence: float

# Initialize FastAPI app
app = FastAPI(
    title="Parliament Pulse - POC",
    description="Local LLM-based email analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize application at startup"""
    logger.info("🏛️  Parliament Pulse POC starting up...")
    logger.info(f"📊 Data directory: {settings.DATA_DIR}")
    logger.info(f"🤖 LLM model: {settings.LLM_MODEL}")
    logger.info(f"🔗 Ollama URL: {settings.OLLAMA_BASE_URL}")
    logger.info("✅ Application ready!")

@app.get("/")
async def root():
    """Root endpoint with POC information"""
    return JSONResponse(
        status_code=200,
        content={
            "name": "Parliament Pulse - POC",
            "version": "0.1.0",
            "description": "Local LLM-based email analysis",
            "status": "ready",
            "endpoints": {
                "analyze": "/analyze - POST text for analysis",
                "stats": "/stats - GET analysis statistics",
                "recent": "/recent - GET recent analyses",
                "health": "/health - System health check",
                "docs": "/docs - API documentation"
            }
        }
    )

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Test LLM connection
        llm_connected = await llm_processor.test_connection()
        
        # Get storage stats
        stats = storage.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "llm_connected": llm_connected,
                "model": settings.LLM_MODEL,
                "ollama_url": settings.OLLAMA_BASE_URL,
                "storage": "connected",
                "analyses_count": stats['total_analyses'],
                "version": "0.1.0"
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

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text using local gpt-oss 20B LLM
    Checks for existing analysis if email_id is provided
    """
    try:
        # Check if this email was already analyzed
        if request.email_id:
            existing_analysis = storage.get_analysis_by_email_id(request.email_id)
            if existing_analysis:
                logger.info(f"Returning cached analysis for email_id: {request.email_id}")
                return AnalyzeResponse(
                    success=True,
                    id=existing_analysis["id"],
                    topic=existing_analysis["topic"],
                    sentiment=existing_analysis["sentiment"],
                    confidence=existing_analysis["confidence"],
                    summary=existing_analysis["summary"],
                    timestamp=existing_analysis["timestamp"],
                    message="Cached analysis (previously analyzed)"
                )
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Starting LLM analysis for ID: {analysis_id}")
        
        # DEBUG: Print what we're sending to LLM
        print(f"DEBUG: Starting analysis for email: {request.subject}")
        print(f"DEBUG: Email text (first 200 chars): {request.text[:200]}...")
        
        # Run LLM analysis
        llm_result = await llm_processor.analyze_email(request.text)
        
        # Create analysis record
        analysis_record = {
            "id": analysis_id,
            "email_id": request.email_id,  # Store the email ID for caching
            "text": request.text,
            "sender_email": request.sender_email,
            "subject": request.subject,
            "topic": llm_result["topic"],
            "sentiment": llm_result["sentiment"],
            "confidence": llm_result["confidence"],
            "summary": llm_result["summary"],
            "timestamp": timestamp,
            "llm_status": llm_result.get("status", "unknown")
        }
        
        # Save to local storage
        storage.add_analysis(analysis_record)
        logger.info(f"Saved LLM analysis: {analysis_id} (status: {llm_result.get('status', 'unknown')})")
        
        # Determine success message
        if llm_result.get("status") == "success":
            message = "LLM analysis completed successfully"
        else:
            message = f"Analysis completed with fallback (LLM status: {llm_result.get('status', 'unknown')})"
        
        return AnalyzeResponse(
            success=True,
            id=analysis_id,
            topic=llm_result["topic"],
            sentiment=llm_result["sentiment"],
            confidence=llm_result["confidence"],
            summary=llm_result["summary"],
            timestamp=timestamp,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/recent")
async def get_recent_analyses(limit: int = 20):
    """Get recent analysis results"""
    try:
        recent = storage.get_recent_analyses(limit=limit)
        return JSONResponse(
            status_code=200,
            content={
                "analyses": recent,
                "count": len(recent)
            }
        )
    except Exception as e:
        logger.error(f"Failed to get recent analyses: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get analysis statistics"""
    try:
        stats = storage.get_stats()
        return StatsResponse(
            total_analyses=stats['total_analyses'],
            topics=stats['topics'],
            sentiments=stats['sentiments'],
            average_confidence=stats['average_confidence']
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/time-series")
async def get_time_series(days: int = 30):
    """Get time-series data for charts"""
    try:
        data = storage.get_time_series_data(days=days)
        return JSONResponse(
            status_code=200,
            content={
                "data": data,
                "days": days,
                "count": len(data)
            }
        )
    except Exception as e:
        logger.error(f"Failed to get time-series data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, debug=settings.DEBUG) 