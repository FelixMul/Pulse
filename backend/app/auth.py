"""
Authentication endpoints for Parliament Pulse
Gmail OAuth 2.0 integration
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from .email_connector import gmail_connector
from .config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

class AuthStatusResponse(BaseModel):
    """Authentication status response model"""
    authenticated: bool
    email: str = None
    profile: Dict[str, Any] = None

class OAuthCallbackRequest(BaseModel):
    """OAuth callback request model"""
    code: str
    state: str

@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status():
    """Get current authentication status"""
    try:
        if gmail_connector.is_authenticated():
            profile = gmail_connector.get_profile()
            return AuthStatusResponse(
                authenticated=True,
                email=profile.get('email'),
                profile=profile
            )
        else:
            return AuthStatusResponse(authenticated=False)
    except Exception as e:
        logger.error(f"Failed to get auth status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get authentication status")

@router.get("/login")
async def start_oauth_flow():
    """Start Gmail OAuth authentication flow"""
    try:
        # Check if already authenticated
        if gmail_connector.is_authenticated():
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Already authenticated",
                    "authenticated": True
                }
            )
        
        # Generate authorization URL
        authorization_url, state = gmail_connector.get_authorization_url()
        
        return JSONResponse(
            status_code=200,
            content={
                "authorization_url": authorization_url,
                "state": state,
                "message": "Visit the authorization URL to complete OAuth flow"
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start OAuth flow: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start OAuth flow")

@router.get("/callback")
async def oauth_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State parameter for security"),
    error: str = Query(None, description="Error parameter if authorization failed")
):
    """Handle OAuth callback from Google"""
    try:
        # Check for authorization errors
        if error:
            logger.warning(f"OAuth authorization failed: {error}")
            return RedirectResponse(
                url=f"{settings.FRONTEND_URL}?auth_error={error}",
                status_code=302
            )
        
        # Exchange authorization code for tokens
        success = gmail_connector.exchange_code_for_tokens(code, state)
        
        if success:
            # Get user profile
            profile = gmail_connector.get_profile()
            logger.info(f"OAuth authentication successful for {profile.get('email')}")
            
            # Redirect to frontend with success
            return RedirectResponse(
                url=f"{settings.FRONTEND_URL}?auth_success=true&email={profile.get('email', '')}",
                status_code=302
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")
    
    except Exception as e:
        logger.error(f"OAuth callback failed: {str(e)}")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}?auth_error=callback_failed",
            status_code=302
        )

@router.post("/callback")
async def oauth_callback_post(request: OAuthCallbackRequest):
    """Handle OAuth callback via POST request (alternative to GET)"""
    try:
        success = gmail_connector.exchange_code_for_tokens(
            request.code, 
            request.state
        )
        
        if success:
            profile = gmail_connector.get_profile()
            logger.info(f"OAuth authentication successful for {profile.get('email')}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Authentication successful",
                    "email": profile.get('email'),
                    "profile": profile
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")
    
    except Exception as e:
        logger.error(f"OAuth callback failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@router.post("/logout")
async def logout():
    """Logout and revoke Gmail access"""
    try:
        success = gmail_connector.disconnect()
        
        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Logout successful"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to logout")
    
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

@router.get("/profile")
async def get_profile():
    """Get authenticated user's Gmail profile"""
    try:
        if not gmail_connector.is_authenticated():
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        profile = gmail_connector.get_profile()
        return JSONResponse(
            status_code=200,
            content={
                "profile": profile,
                "authenticated": True
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to get profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get profile")

@router.get("/test")
async def test_gmail_connection():
    """Test Gmail API connection (for development/debugging)"""
    try:
        if not gmail_connector.is_authenticated():
            raise HTTPException(status_code=401, detail="Not authenticated with Gmail")
        
        # Try to fetch a small number of emails to test connection
        result = gmail_connector.fetch_emails(max_results=1)
        
        return JSONResponse(
            status_code=200,
            content={
                "connection": "ok",
                "test_result": {
                    "can_fetch_emails": True,
                    "sample_count": len(result.get('messages', [])),
                    "total_estimate": result.get('result_size_estimate', 0)
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Gmail connection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}") 