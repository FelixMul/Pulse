"""
Configuration settings for Parliament Pulse POC
Simplified configuration for local development
"""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for the POC"""
    
    # Server Configuration
    HOST: str = Field(default="127.0.0.1", description="Server host")
    PORT: int = Field(default=8080, description="Server port")
    DEBUG: bool = Field(default=True, description="Debug mode")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],  # Allow all origins for POC
        description="CORS allowed origins"
    )
    
    # Data Directory for local storage
    DATA_DIR: str = Field(
        default="./data",
        description="Directory for storing data and analysis results"
    )
    
    # LLM Configuration
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    LLM_MODEL: str = Field(
        default="gpt-oss:20b",
        description="LLM model to use for analysis"
    )
    LLM_TIMEOUT: int = Field(
        default=60,
        description="LLM request timeout in seconds"
    )
    
    # Analysis Configuration
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum confidence threshold for analysis results"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = "ignore"
    
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