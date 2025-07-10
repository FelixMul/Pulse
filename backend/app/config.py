"""
Configuration settings for Parliament Pulse
Environment variables and application configuration
"""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Server Configuration
    HOST: str = Field(default="127.0.0.1", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=True, description="Debug mode")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="CORS allowed origins"
    )
    
    # Frontend Configuration
    FRONTEND_URL: str = Field(
        default="http://127.0.0.1:3000",
        description="Frontend URL for OAuth redirects"
    )
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="sqlite:///./data/emails.db",
        description="Database connection URL"
    )
    
    # Data Directory for ML models and processed data
    DATA_DIR: str = Field(
        default="./data",
        description="Directory for storing data and ML models"
    )
    
    # Google OAuth Configuration
    GOOGLE_CLIENT_ID: str = Field(
        default="",
        description="Google OAuth client ID"
    )
    GOOGLE_CLIENT_SECRET: str = Field(
        default="",
        description="Google OAuth client secret"
    )
    GOOGLE_REDIRECT_URI: str = Field(
        default="http://127.0.0.1:8000/api/auth/callback",
        description="OAuth redirect URI"
    )
    GOOGLE_SCOPES: List[str] = Field(
        default=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"
        ],
        description="Google API scopes"
    )
    
    # NLP Configuration
    NLP_BATCH_SIZE: int = Field(default=50, description="Batch size for NLP processing")
    TOPIC_MODEL_MIN_DOCS: int = Field(default=5, description="Minimum documents needed for topic modeling")
    SPAM_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Spam detection confidence threshold")
    
    # Email Processing Configuration
    EMAIL_FETCH_LIMIT: int = Field(default=100, description="Maximum emails to fetch per request")
    EMAIL_PROCESSING_ENABLED: bool = Field(default=True, description="Enable email processing")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables
    
    def validate_oauth_settings(self) -> bool:
        """Validate that OAuth settings are properly configured"""
        return bool(
            self.GOOGLE_CLIENT_ID and 
            self.GOOGLE_CLIENT_SECRET and 
            self.GOOGLE_REDIRECT_URI
        )
    
    def validate_database_settings(self) -> bool:
        """Validate database configuration"""
        return bool(self.DATABASE_URL)
    
    def ensure_data_directory(self) -> bool:
        """Ensure data directory exists"""
        try:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            return True
        except Exception as e:
            print(f"Failed to create data directory: {e}")
            return False

# Create global settings instance
settings = Settings()

# Ensure data directory exists on startup
settings.ensure_data_directory() 