"""
Simple local JSON storage for Parliament Pulse POC
Replaces complex database with file-based storage
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import settings


class LocalStorage:
    """Simple JSON-based storage for analysis results"""
    
    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR)
        self.analyses_file = self.data_dir / "analyses.json"
        self.ensure_files_exist()
    
    def ensure_files_exist(self):
        """Ensure storage files exist"""
        self.data_dir.mkdir(exist_ok=True)
        
        if not self.analyses_file.exists():
            self.save_analyses([])
    
    def save_analyses(self, analyses: List[Dict[str, Any]]):
        """Save all analyses to JSON file"""
        try:
            with open(self.analyses_file, 'w', encoding='utf-8') as f:
                json.dump(analyses, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Failed to save analyses: {e}")
    
    def load_analyses(self) -> List[Dict[str, Any]]:
        """Load all analyses from JSON file"""
        try:
            if self.analyses_file.exists():
                with open(self.analyses_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load analyses: {e}")
        return []
    
    def add_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Add a single analysis result"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in analysis:
                analysis['timestamp'] = datetime.now().isoformat()
            
            # Add unique ID if not present
            if 'id' not in analysis:
                analysis['id'] = f"analysis_{len(self.load_analyses()) + 1}_{int(datetime.now().timestamp())}"
            
            analyses = self.load_analyses()
            analyses.append(analysis)
            self.save_analyses(analyses)
            return True
        except Exception as e:
            print(f"Failed to add analysis: {e}")
            return False
    
    def get_recent_analyses(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent analyses"""
        analyses = self.load_analyses()
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return analyses[:limit]
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get specific analysis by ID"""
        analyses = self.load_analyses()
        for analysis in analyses:
            if analysis.get('id') == analysis_id:
                return analysis
        return None
    
    def get_analysis_by_email_id(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Get specific analysis by email ID (for caching)"""
        if not email_id:
            return None
        analyses = self.load_analyses()
        for analysis in analyses:
            if analysis.get('email_id') == email_id:
                return analysis
        return None
    
    def get_time_series_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get time-series data for charts"""
        analyses = self.load_analyses()
        
        # Filter recent analyses
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_analyses = []
        for analysis in analyses:
            try:
                analysis_date = datetime.fromisoformat(analysis.get('timestamp', ''))
                if analysis_date >= cutoff_date:
                    recent_analyses.append(analysis)
            except ValueError:
                continue  # Skip invalid timestamps
        
        return recent_analyses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics"""
        analyses = self.load_analyses()
        
        if not analyses:
            return {
                'total_analyses': 0,
                'topics': {},
                'sentiments': {},
                'average_confidence': 0.0
            }
        
        # Count topics and sentiments
        topics = {}
        sentiments = {}
        confidences = []
        
        for analysis in analyses:
            # Count topics
            topic = analysis.get('topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
            
            # Count sentiments
            sentiment = analysis.get('sentiment', 'unknown')
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
            
            # Collect confidence scores
            confidence = analysis.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'total_analyses': len(analyses),
            'topics': topics,
            'sentiments': sentiments,
            'average_confidence': avg_confidence
        }
    
    def clear_all_data(self) -> bool:
        """Clear all stored data (for testing)"""
        try:
            self.save_analyses([])
            return True
        except Exception as e:
            print(f"Failed to clear data: {e}")
            return False


# Global storage instance
storage = LocalStorage()
