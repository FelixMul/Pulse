"""
Gmail OAuth Connector for Parliament Pulse
Secure email access using OAuth 2.0 without password storage
"""

import os
import json
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import base64
import email
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import keyring

from .config import settings

logger = logging.getLogger(__name__)

class GmailConnector:
    """Handles Gmail OAuth authentication and email fetching"""
    
    def __init__(self):
        self.credentials: Optional[Credentials] = None
        self.service = None
        self.app_name = "ParliamentPulse"
        
    def _get_credentials_from_keyring(self) -> Optional[Credentials]:
        """Retrieve stored credentials from system keyring"""
        try:
            creds_json = keyring.get_password(self.app_name, "gmail_credentials")
            if creds_json:
                creds_data = json.loads(creds_json)
                credentials = Credentials.from_authorized_user_info(creds_data)
                
                # Refresh if expired
                if credentials.expired and credentials.refresh_token:
                    try:
                        credentials.refresh(Request())
                        # Save refreshed credentials
                        self._save_credentials_to_keyring(credentials)
                        logger.info("Gmail credentials refreshed successfully")
                    except Exception as e:
                        logger.error(f"Failed to refresh credentials: {str(e)}")
                        return None
                
                return credentials
        except Exception as e:
            logger.error(f"Failed to retrieve credentials from keyring: {str(e)}")
        return None
    
    def _save_credentials_to_keyring(self, credentials: Credentials) -> bool:
        """Save credentials to system keyring"""
        try:
            creds_json = credentials.to_json()
            keyring.set_password(self.app_name, "gmail_credentials", creds_json)
            logger.info("Gmail credentials saved to keyring successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save credentials to keyring: {str(e)}")
            return False
    
    def _delete_credentials_from_keyring(self) -> bool:
        """Delete stored credentials from keyring"""
        try:
            keyring.delete_password(self.app_name, "gmail_credentials")
            logger.info("Gmail credentials deleted from keyring")
            return True
        except Exception as e:
            logger.error(f"Failed to delete credentials from keyring: {str(e)}")
            return False
    
    def get_authorization_url(self) -> tuple[str, str]:
        """
        Generate authorization URL for OAuth flow
        Returns: (authorization_url, state)
        """
        if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
            raise ValueError("Google OAuth credentials not configured")
        
        # Create OAuth flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [settings.GOOGLE_REDIRECT_URI]
                }
            },
            scopes=settings.GOOGLE_SCOPES
        )
        flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
        
        # Generate authorization URL
        authorization_url, state = flow.authorization_url(
            access_type='offline',  # Enable refresh tokens
            include_granted_scopes='true',  # Incremental authorization
            prompt='consent'  # Force consent screen to get refresh token
        )
        
        # Store state in keyring temporarily
        keyring.set_password(self.app_name, "oauth_state", state)
        
        return authorization_url, state
    
    def exchange_code_for_tokens(self, authorization_code: str, state: str) -> bool:
        """
        Exchange authorization code for access tokens
        Returns: True if successful, False otherwise
        """
        try:
            # Verify state parameter
            stored_state = keyring.get_password(self.app_name, "oauth_state")
            if stored_state != state:
                raise ValueError("Invalid state parameter")
            
            # Delete used state
            keyring.delete_password(self.app_name, "oauth_state")
            
            # Create OAuth flow
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": settings.GOOGLE_CLIENT_ID,
                        "client_secret": settings.GOOGLE_CLIENT_SECRET,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [settings.GOOGLE_REDIRECT_URI]
                    }
                },
                scopes=settings.GOOGLE_SCOPES
            )
            flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
            
            # Exchange code for tokens
            flow.fetch_token(code=authorization_code)
            credentials = flow.credentials
            
            # Save credentials
            if self._save_credentials_to_keyring(credentials):
                self.credentials = credentials
                logger.info("Gmail OAuth setup completed successfully")
                return True
            
        except Exception as e:
            logger.error(f"Failed to exchange authorization code: {str(e)}")
        
        return False
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated with Gmail"""
        if not self.credentials:
            self.credentials = self._get_credentials_from_keyring()
        
        return self.credentials is not None and self.credentials.valid
    
    def disconnect(self) -> bool:
        """Disconnect from Gmail and delete stored credentials"""
        try:
            # Revoke tokens if possible
            if self.credentials and self.credentials.token:
                try:
                    # Revoke the token
                    import requests
                    requests.post(
                        'https://oauth2.googleapis.com/revoke',
                        params={'token': self.credentials.token},
                        headers={'content-type': 'application/x-www-form-urlencoded'}
                    )
                except Exception as e:
                    logger.warning(f"Failed to revoke token: {str(e)}")
            
            # Delete stored credentials
            self._delete_credentials_from_keyring()
            self.credentials = None
            self.service = None
            
            logger.info("Disconnected from Gmail successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from Gmail: {str(e)}")
            return False
    
    def _get_service(self):
        """Get authenticated Gmail service"""
        if not self.is_authenticated():
            raise ValueError("Not authenticated with Gmail")
        
        if not self.service:
            self.service = build('gmail', 'v1', credentials=self.credentials)
        
        return self.service
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user's Gmail profile information"""
        try:
            service = self._get_service()
            profile = service.users().getProfile(userId='me').execute()
            
            return {
                'email': profile.get('emailAddress'),
                'messages_total': profile.get('messagesTotal', 0),
                'threads_total': profile.get('threadsTotal', 0),
                'history_id': profile.get('historyId')
            }
        except Exception as e:
            logger.error(f"Failed to get Gmail profile: {str(e)}")
            raise
    
    def fetch_emails(self, 
                    max_results: int = 100, 
                    query: str = "", 
                    page_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch emails from Gmail
        
        Args:
            max_results: Maximum number of emails to fetch
            query: Gmail search query (e.g., "is:unread", "from:example@gmail.com")
            page_token: Token for pagination
            
        Returns:
            Dictionary with messages and next page token
        """
        try:
            service = self._get_service()
            
            # Build request parameters
            params = {
                'userId': 'me',
                'maxResults': min(max_results, 500),  # Gmail API limit
                'q': query
            }
            if page_token:
                params['pageToken'] = page_token
            
            # Get message list
            result = service.users().messages().list(**params).execute()
            messages = result.get('messages', [])
            
            # Fetch full message details
            detailed_messages = []
            for message in messages:
                try:
                    msg_detail = service.users().messages().get(
                        userId='me', 
                        id=message['id'],
                        format='full'
                    ).execute()
                    detailed_messages.append(msg_detail)
                except Exception as e:
                    logger.warning(f"Failed to fetch message {message['id']}: {str(e)}")
                    continue
            
            return {
                'messages': detailed_messages,
                'next_page_token': result.get('nextPageToken'),
                'result_size_estimate': result.get('resultSizeEstimate', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch emails: {str(e)}")
            raise
    
    def parse_email_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Gmail API message into structured format
        
        Returns:
            Dictionary with parsed email data
        """
        try:
            headers = {h['name']: h['value'] for h in message['payload'].get('headers', [])}
            
            # Extract basic information
            email_data = {
                'id': message['id'],
                'thread_id': message['threadId'],
                'label_ids': message.get('labelIds', []),
                'snippet': message.get('snippet', ''),
                'history_id': message.get('historyId'),
                'internal_date': message.get('internalDate'),
                'size_estimate': message.get('sizeEstimate', 0)
            }
            
            # Parse headers
            email_data.update({
                'sender_email': headers.get('From', ''),
                'sender_name': self._extract_name_from_email(headers.get('From', '')),
                'recipient_email': headers.get('To', ''),
                'subject': headers.get('Subject', ''),
                'date': headers.get('Date', ''),
                'received_at': self._parse_email_date(headers.get('Date', '')),
                'message_id': headers.get('Message-ID', ''),
                'in_reply_to': headers.get('In-Reply-To', ''),
                'references': headers.get('References', '')
            })
            
            # Extract body
            email_data['raw_body'] = self._extract_email_body(message['payload'])
            
            return email_data
            
        except Exception as e:
            logger.error(f"Failed to parse email message: {str(e)}")
            raise
    
    def _extract_name_from_email(self, from_header: str) -> str:
        """Extract name from email 'From' header"""
        try:
            if '<' in from_header:
                name_part = from_header.split('<')[0].strip()
                return name_part.strip('"\'')
            return ""
        except:
            return ""
    
    def _parse_email_date(self, date_str: str) -> Optional[datetime]:
        """Parse email date string to datetime"""
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            return None
    
    def _extract_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract email body from Gmail API payload"""
        try:
            body = ""
            
            if 'parts' in payload:
                # Multipart message
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        if 'data' in part['body']:
                            body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    elif part['mimeType'] == 'text/html' and not body:
                        # Fallback to HTML if no plain text
                        if 'data' in part['body']:
                            html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                            # TODO: Convert HTML to plain text
                            body = html_body
            else:
                # Single part message
                if payload['mimeType'] == 'text/plain' and 'data' in payload['body']:
                    body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
                elif payload['mimeType'] == 'text/html' and 'data' in payload['body']:
                    html_body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
                    # TODO: Convert HTML to plain text
                    body = html_body
            
            return body
            
        except Exception as e:
            logger.warning(f"Failed to extract email body: {str(e)}")
            return ""

# Global Gmail connector instance
gmail_connector = GmailConnector() 