#!/usr/bin/env python3
"""
Parliament Pulse - Unified Server Startup
Runs both backend API and frontend static server in a single process
"""

import asyncio
import multiprocessing
import os
import sys
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def create_app():
    """Create the unified FastAPI application"""
    # Import the main app from backend
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    from app.main import app as backend_app
    
    # Create a new FastAPI app that includes both backend and static files
    app = FastAPI(
        title="Parliament Pulse - Unified Server",
        description="Local email analysis with integrated frontend",
        version="0.1.0"
    )
    
    # Mount the backend API
    app.mount("/api", backend_app, name="backend")
    
    # Add a simple endpoint to serve CSV data BEFORE mounting static files
    @app.get("/data/{filename}")
    async def serve_csv(filename: str):
        """Serve CSV files from the data directory"""
        data_path = Path(__file__).parent / "data" / filename
        if data_path.exists() and data_path.suffix == '.csv':
            from fastapi.responses import FileResponse
            return FileResponse(data_path, media_type='text/csv')
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="File not found")
    
    # Mount static files for test-ui (mount last so it doesn't override other routes)
    test_ui_path = Path(__file__).parent / "test-ui"
    if test_ui_path.exists():
        app.mount("/", StaticFiles(directory=str(test_ui_path), html=True), name="static")
    
    return app


def open_browser_after_delay():
    """Open browser after server starts"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://localhost:8080")


def main():
    """Main entry point"""
    print("🏛️  Parliament Pulse - Local Email Analysis")
    print("=" * 50)
    print("🚀 Starting unified server...")
    print("📡 Backend API: http://localhost:8080/api")
    print("🌐 Frontend UI: http://localhost:8080")
    print("📚 API docs: http://localhost:8080/api/docs")
    print("=" * 50)
    
    # Create the unified app
    app = create_app()
    
    # Start browser in background
    browser_process = multiprocessing.Process(target=open_browser_after_delay)
    browser_process.start()
    
    try:
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            access_log=False
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Parliament Pulse...")
    except OSError as e:
        if "address already in use" in str(e).lower():
            print("\n❌ Error: Port 8080 is already in use!")
            print("💡 Try running: lsof -ti:8080 | xargs kill -9")
            print("   Then restart with: uv run python start_server.py")
        else:
            print(f"\n❌ Server error: {e}")
    finally:
        browser_process.terminate()


if __name__ == "__main__":
    main()
