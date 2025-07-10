"""
Database management for Parliament Pulse
SQLite database with email storage and analytics
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from .config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for email storage and analytics"""
    
    def __init__(self):
        self.db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        self.init_database()
    
    def get_connection(self):
        """Get database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable accessing columns by name
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create emails table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS emails (
                        id TEXT PRIMARY KEY,
                        email_id TEXT UNIQUE NOT NULL,
                        sender_email TEXT NOT NULL,
                        sender_name TEXT,
                        subject TEXT,
                        received_at DATETIME,
                        raw_body TEXT,
                        cleaned_body TEXT,
                        topic TEXT,
                        topic_id INTEGER,
                        sentiment TEXT,
                        sentiment_compound REAL,
                        sentiment_positive REAL,
                        sentiment_negative REAL,
                        sentiment_neutral REAL,
                        is_spam BOOLEAN DEFAULT FALSE,
                        spam_confidence REAL,
                        processed_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create topics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS topics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_id INTEGER UNIQUE NOT NULL,
                        topic_label TEXT NOT NULL,
                        topic_words TEXT,
                        email_count INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create processing_status table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_status (
                        id INTEGER PRIMARY KEY,
                        last_sync_at DATETIME,
                        emails_processed INTEGER DEFAULT 0,
                        last_email_date DATETIME,
                        nlp_model_version TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert default processing status if not exists
                cursor.execute("""
                    INSERT OR IGNORE INTO processing_status (id, emails_processed)
                    VALUES (1, 0)
                """)
                
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_received_at ON emails(received_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender_email)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_topic ON emails(topic_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_sentiment ON emails(sentiment)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_spam ON emails(is_spam)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def insert_email(self, email_data: Dict[str, Any]) -> bool:
        """Insert a single email into the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO emails (
                        id, email_id, sender_email, sender_name, subject,
                        received_at, raw_body, cleaned_body, topic, topic_id,
                        sentiment, sentiment_compound, sentiment_positive,
                        sentiment_negative, sentiment_neutral, is_spam,
                        spam_confidence, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    email_data.get('id'),
                    email_data.get('email_id'),
                    email_data.get('sender_email'),
                    email_data.get('sender_name'),
                    email_data.get('subject'),
                    email_data.get('received_at'),
                    email_data.get('raw_body'),
                    email_data.get('cleaned_body'),
                    email_data.get('topic'),
                    email_data.get('topic_id'),
                    email_data.get('sentiment'),
                    email_data.get('sentiment_compound'),
                    email_data.get('sentiment_positive'),
                    email_data.get('sentiment_negative'),
                    email_data.get('sentiment_neutral'),
                    email_data.get('is_spam'),
                    email_data.get('spam_confidence'),
                    email_data.get('processed_at')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert email: {str(e)}")
            return False
    
    def get_email_count(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> int:
        """Get total email count with optional date filtering"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT COUNT(*) FROM emails WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                cursor.execute(query, params)
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get email count: {str(e)}")
            return 0
    
    def get_processed_email_count(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> int:
        """Get count of processed emails (with sentiment analysis)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT COUNT(*) FROM emails WHERE sentiment IS NOT NULL"
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                cursor.execute(query, params)
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get processed email count: {str(e)}")
            return 0
    
    def get_spam_count(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> int:
        """Get count of spam emails"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT COUNT(*) FROM emails WHERE is_spam = 1"
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                cursor.execute(query, params)
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get spam count: {str(e)}")
            return 0
    
    def get_sentiment_statistics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, float]:
        """Get sentiment statistics including average scores"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        AVG(sentiment_compound) as avg_compound,
                        AVG(sentiment_positive) as avg_positive,
                        AVG(sentiment_negative) as avg_negative,
                        AVG(sentiment_neutral) as avg_neutral
                    FROM emails 
                    WHERE sentiment_compound IS NOT NULL
                """
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                return {
                    'average_compound': row[0] or 0.0,
                    'average_positive': row[1] or 0.0,
                    'average_negative': row[2] or 0.0,
                    'average_neutral': row[3] or 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get sentiment statistics: {str(e)}")
            return {'average_compound': 0.0, 'average_positive': 0.0, 'average_negative': 0.0, 'average_neutral': 0.0}
    
    def get_sentiment_breakdown(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, int]:
        """Get count of emails by sentiment"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT sentiment, COUNT(*) 
                    FROM emails 
                    WHERE sentiment IS NOT NULL
                """
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                query += " GROUP BY sentiment"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
                for row in results:
                    sentiment_counts[row[0]] = row[1]
                
                return sentiment_counts
                
        except Exception as e:
            logger.error(f"Failed to get sentiment breakdown: {str(e)}")
            return {'positive': 0, 'negative': 0, 'neutral': 0}
    
    def get_topic_statistics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get topic statistics with email counts"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT topic, topic_id, COUNT(*) as count
                    FROM emails 
                    WHERE topic IS NOT NULL AND topic != 'Unknown' AND topic != 'Outlier'
                """
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                query += " GROUP BY topic, topic_id ORDER BY count DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                topics = []
                for row in results:
                    topics.append({
                        'topic_label': row[0],
                        'topic_id': row[1],
                        'count': row[2]
                    })
                
                return topics
                
        except Exception as e:
            logger.error(f"Failed to get topic statistics: {str(e)}")
            return []
    
    def get_sentiment_by_topic(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sentiment breakdown by topic"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        topic,
                        sentiment,
                        COUNT(*) as count,
                        AVG(sentiment_compound) as avg_compound
                    FROM emails 
                    WHERE topic IS NOT NULL AND sentiment IS NOT NULL 
                    AND topic != 'Unknown' AND topic != 'Outlier'
                """
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                query += " GROUP BY topic, sentiment ORDER BY topic, sentiment"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Group by topic
                topics = {}
                for row in results:
                    topic = row[0]
                    if topic not in topics:
                        topics[topic] = {
                            'topic_label': topic,
                            'sentiments': {},
                            'total_count': 0
                        }
                    
                    topics[topic]['sentiments'][row[1]] = {
                        'count': row[2],
                        'avg_compound': row[3] or 0.0
                    }
                    topics[topic]['total_count'] += row[2]
                
                # Convert to list and limit
                topic_list = list(topics.values())
                topic_list.sort(key=lambda x: x['total_count'], reverse=True)
                
                return topic_list[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get sentiment by topic: {str(e)}")
            return []
    
    def get_email_volume_trend(self, granularity: str = "week", start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get email volume trend over time"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Determine date formatting based on granularity
                if granularity == "day":
                    date_format = "%Y-%m-%d"
                    date_trunc = "date(received_at)"
                elif granularity == "month":
                    date_format = "%Y-%m"
                    date_trunc = "strftime('%Y-%m', received_at)"
                else:  # week
                    date_format = "%Y-%W"
                    date_trunc = "strftime('%Y-%W', received_at)"
                
                query = f"""
                    SELECT 
                        {date_trunc} as period,
                        COUNT(*) as count
                    FROM emails 
                    WHERE received_at IS NOT NULL
                """
                params = []
                
                if start_date:
                    query += " AND received_at >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND received_at <= ?"
                    params.append(end_date.isoformat())
                
                query += f" GROUP BY {date_trunc} ORDER BY period"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                volume_data = []
                for row in results:
                    if granularity == "week":
                        # Convert week format to actual date
                        year, week = row[0].split('-')
                        date_obj = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
                    elif granularity == "month":
                        date_obj = datetime.strptime(row[0], "%Y-%m")
                    else:  # day
                        date_obj = datetime.strptime(row[0], "%Y-%m-%d")
                    
                    volume_data.append({
                        'date': date_obj,
                        'count': row[1]
                    })
                
                return volume_data
                
        except Exception as e:
            logger.error(f"Failed to get email volume trend: {str(e)}")
            return []
    
    def get_recent_emails(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent emails with basic information"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        id, sender_email, sender_name, subject, received_at,
                        sentiment, is_spam, topic
                    FROM emails 
                    ORDER BY received_at DESC 
                    LIMIT ?
                """, (limit,))
                
                results = cursor.fetchall()
                
                emails = []
                for row in results:
                    emails.append({
                        'id': row[0],
                        'sender_email': row[1],
                        'sender_name': row[2],
                        'subject': row[3],
                        'received_at': row[4],
                        'sentiment': row[5],
                        'is_spam': bool(row[6]),
                        'topic': row[7]
                    })
                
                return emails
                
        except Exception as e:
            logger.error(f"Failed to get recent emails: {str(e)}")
            return []
    
    def get_processing_status(self) -> Optional[Dict[str, Any]]:
        """Get processing status information"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        last_sync_at, emails_processed, last_email_date,
                        nlp_model_version, updated_at
                    FROM processing_status 
                    WHERE id = 1
                """)
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': 1,
                        'last_sync_at': row[0],
                        'emails_processed': row[1],
                        'last_email_date': row[2],
                        'nlp_model_version': row[3],
                        'updated_at': row[4]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get processing status: {str(e)}")
            return None
    
    def update_processing_status(self, **kwargs) -> bool:
        """Update processing status"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build update query dynamically
                update_fields = []
                params = []
                
                for field, value in kwargs.items():
                    if field in ['last_sync_at', 'emails_processed', 'last_email_date', 'nlp_model_version']:
                        update_fields.append(f"{field} = ?")
                        params.append(value)
                
                if update_fields:
                    update_fields.append("updated_at = CURRENT_TIMESTAMP")
                    params.append(1)  # for WHERE id = 1
                    
                    query = f"UPDATE processing_status SET {', '.join(update_fields)} WHERE id = ?"
                    cursor.execute(query, params)
                    conn.commit()
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update processing status: {str(e)}")
            return False

# Global database manager instance
db_manager = DatabaseManager() 