#!/usr/bin/env python3
"""
Development server runner for Parliament Pulse
Run this script from the backend/ directory to start the FastAPI server
"""

import sys
import os

# Add the current directory to Python path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from app.main import app
    from app.config import settings
    import uvicorn
    
    print("🏛️  Parliament Pulse - Email Analysis for Parliamentarians")
    print("=" * 55)
    print(f"🚀 Starting development server...")
    print(f"📡 API will be available at: http://{settings.HOST}:{settings.PORT}")
    print(f"📚 API docs will be available at: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"🔧 Debug mode: {settings.DEBUG}")
    print(f"🔐 OAuth configured: {settings.validate_oauth_settings()}")
    print("=" * 55)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    ) 