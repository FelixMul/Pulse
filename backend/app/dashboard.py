"""
Dashboard API endpoints for Parliament Pulse
Provides analytics data for the frontend dashboard
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .database import db_manager
from .email_connector import gmail_connector
from .nlp_processor import nlp_processor
from .config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

class DashboardStatsResponse(BaseModel):
    """Dashboard statistics response model"""
    total_emails: int
    emails_processed: int
    average_sentiment: float
    unique_topics: int
    spam_count: int
    date_range: Dict[str, str]

class EmailVolumeResponse(BaseModel):
    """Email volume trend response model"""
    labels: List[str]
    data: List[int]
    granularity: str

class TopicBreakdownResponse(BaseModel):
    """Topic breakdown response model"""
    topics: List[Dict[str, Any]]
    total_processed: int

class SentimentAnalysisResponse(BaseModel):
    """Sentiment analysis response model"""
    sentiment_counts: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    average_compound: float

@router.get("/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get overall dashboard statistics"""
    try:
        # Parse date filters
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get email statistics
        total_emails = db_manager.get_email_count(start_date=start_dt, end_date=end_dt)
        emails_processed = db_manager.get_processed_email_count(start_date=start_dt, end_date=end_dt)
        
        # Get sentiment statistics
        sentiment_stats = db_manager.get_sentiment_statistics(start_date=start_dt, end_date=end_dt)
        average_sentiment = sentiment_stats.get('average_compound', 0.0)
        
        # Get topic statistics
        topic_stats = db_manager.get_topic_statistics(start_date=start_dt, end_date=end_dt)
        unique_topics = len(topic_stats)
        
        # Get spam statistics
        spam_count = db_manager.get_spam_count(start_date=start_dt, end_date=end_dt)
        
        # Date range info
        date_range = {
            "start": start_date or "all",
            "end": end_date or "all"
        }
        
        return DashboardStatsResponse(
            total_emails=total_emails,
            emails_processed=emails_processed,
            average_sentiment=average_sentiment,
            unique_topics=unique_topics,
            spam_count=spam_count,
            date_range=date_range
        )
        
    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard statistics")

@router.get("/email-volume", response_model=EmailVolumeResponse)
async def get_email_volume_trend(
    granularity: str = Query("week", enum=["day", "week", "month"], description="Time granularity"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get email volume trend over time"""
    try:
        # Parse date filters
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get volume data from database
        volume_data = db_manager.get_email_volume_trend(
            granularity=granularity,
            start_date=start_dt,
            end_date=end_dt
        )
        
        # Format for chart
        labels = []
        data = []
        
        for item in volume_data:
            if granularity == "day":
                labels.append(item['date'].strftime("%Y-%m-%d"))
            elif granularity == "week":
                labels.append(f"Week of {item['date'].strftime('%Y-%m-%d')}")
            else:  # month
                labels.append(item['date'].strftime("%Y-%m"))
            data.append(item['count'])
        
        return EmailVolumeResponse(
            labels=labels,
            data=data,
            granularity=granularity
        )
        
    except Exception as e:
        logger.error(f"Failed to get email volume trend: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve email volume data")

@router.get("/topics", response_model=TopicBreakdownResponse)
async def get_topic_breakdown(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of topics to return"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get topic breakdown with email counts"""
    try:
        # Parse date filters
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get topic statistics from database
        topic_stats = db_manager.get_topic_statistics(
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        # Get total processed emails for percentage calculation
        total_processed = sum(topic['count'] for topic in topic_stats)
        
        # Add percentage to each topic
        for topic in topic_stats:
            topic['percentage'] = (topic['count'] / max(total_processed, 1)) * 100
        
        return TopicBreakdownResponse(
            topics=topic_stats,
            total_processed=total_processed
        )
        
    except Exception as e:
        logger.error(f"Failed to get topic breakdown: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve topic data")

@router.get("/sentiment", response_model=SentimentAnalysisResponse)
async def get_sentiment_analysis(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get sentiment analysis breakdown"""
    try:
        # Parse date filters
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get sentiment statistics from database
        sentiment_data = db_manager.get_sentiment_breakdown(
            start_date=start_dt,
            end_date=end_dt
        )
        
        # Calculate totals and percentages
        total_emails = sum(sentiment_data.values())
        sentiment_percentages = {}
        
        for sentiment, count in sentiment_data.items():
            sentiment_percentages[sentiment] = (count / max(total_emails, 1)) * 100
        
        # Get average sentiment score
        sentiment_stats = db_manager.get_sentiment_statistics(
            start_date=start_dt,
            end_date=end_dt
        )
        average_compound = sentiment_stats.get('average_compound', 0.0)
        
        return SentimentAnalysisResponse(
            sentiment_counts=sentiment_data,
            sentiment_percentages=sentiment_percentages,
            average_compound=average_compound
        )
        
    except Exception as e:
        logger.error(f"Failed to get sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sentiment data")

@router.get("/sentiment-by-topic")
async def get_sentiment_by_topic(
    limit: int = Query(10, ge=1, le=20, description="Maximum number of topics to return"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get sentiment breakdown by topic"""
    try:
        # Parse date filters
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get sentiment by topic data
        sentiment_by_topic = db_manager.get_sentiment_by_topic(
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "topics": sentiment_by_topic,
                "total_topics": len(sentiment_by_topic)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get sentiment by topic: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sentiment by topic data")

@router.get("/spam-statistics")
async def get_spam_statistics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get spam detection statistics"""
    try:
        # Parse date filters
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get spam statistics
        total_emails = db_manager.get_email_count(start_date=start_dt, end_date=end_dt)
        spam_count = db_manager.get_spam_count(start_date=start_dt, end_date=end_dt)
        legitimate_count = total_emails - spam_count
        
        spam_percentage = (spam_count / max(total_emails, 1)) * 100
        legitimate_percentage = (legitimate_count / max(total_emails, 1)) * 100
        
        return JSONResponse(
            status_code=200,
            content={
                "total_emails": total_emails,
                "spam_count": spam_count,
                "legitimate_count": legitimate_count,
                "spam_percentage": spam_percentage,
                "legitimate_percentage": legitimate_percentage
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get spam statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve spam statistics")

@router.get("/recent-activity")
async def get_recent_activity(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of recent emails to return")
):
    """Get recent email activity"""
    try:
        recent_emails = db_manager.get_recent_emails(limit=limit)
        
        return JSONResponse(
            status_code=200,
            content={
                "emails": recent_emails,
                "count": len(recent_emails)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get recent activity: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent activity")

@router.get("/processing-status")
async def get_processing_status():
    """Get email processing status and system health"""
    try:
        # Get processing status from database
        processing_status = db_manager.get_processing_status()
        
        # Get NLP model status
        nlp_status = nlp_processor.get_model_status()
        
        # Get authentication status
        auth_status = gmail_connector.is_authenticated()
        
        return JSONResponse(
            status_code=200,
            content={
                "processing_status": processing_status,
                "nlp_models": nlp_status,
                "gmail_authenticated": auth_status,
                "system_ready": all([
                    processing_status is not None,
                    nlp_status['sentiment_analyzer']['ready'],
                    nlp_status['spam_detector']['trained']
                ])
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get processing status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve processing status") 