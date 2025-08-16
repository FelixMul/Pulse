"""
LLM Processor for Parliament Pulse POC
Uses gpt-oss 20B via Ollama for topic extraction and sentiment analysis
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
import httpx

from .config import settings

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Handles LLM-based analysis using gpt-oss 20B"""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT
        self._last_request_time = 0
        self._min_request_interval = 0.5  # Minimum 0.5 seconds between requests
        
        # Define the comprehensive topic categories from the training data
        self.topics = [
            "Healthcare & NHS", "Housing & Planning", "Immigration & Asylum", 
            "Social Security & Benefits", "Cost of Living & Economy", "Education & Schools",
            "Transportation & Infrastructure", "Environment & Climate Change", 
            "Crime & Community Safety", "Employment & Workers' Rights", 
            "Local Government & Council Tax", "Taxation & Public Spending",
            "Pensions & National Insurance", "Digital & Technology", 
            "Agriculture & Rural Affairs", "Business & Enterprise",
            "Foreign Affairs & International Development", "Culture, Media & Sport",
            "Justice & Legal System", "Defence & National Security", 
            "Energy & Utilities", "Consumer Rights & Issues", "Mental Health Services",
            "Childcare & Family Support", "Disability Rights & Access", "Animal Welfare",
            "Trade & Brexit Issues", "Planning & Development", "Local Campaign Support"
        ]
        
        # Define sentiment categories
        self.sentiments = [
            "very_negative", "negative", "neutral", "positive", "very_positive"
        ]
    
    async def analyze_email(self, email_text: str) -> Dict[str, Any]:
        """
        Analyze email using gpt-oss 20B for topic and sentiment
        Returns structured JSON with topic, sentiment, confidence, and summary
        """
        try:
            # Create structured prompt
            prompt = self._create_analysis_prompt(email_text)
            
            # Call Ollama API
            analysis_result = await self._call_ollama(prompt)
            
            if analysis_result:
                # Validate and return result
                return self._validate_analysis_result(analysis_result)
            else:
                return self._create_fallback_result("LLM call failed")
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return self._create_fallback_result(f"Error: {str(e)}")
    
    def _create_analysis_prompt(self, email_text: str) -> str:
        """Create structured prompt for email analysis"""
        
        original_length = len(email_text)
        
        # Increase limit significantly - most emails are under 4000 chars
        # Keep more content for better analysis
        if len(email_text) > 4000:
            email_text = email_text[:4000] + "..."
            print(f"DEBUG: Email truncated from {original_length} to 4000 characters")
        else:
            print(f"DEBUG: Email sent in full ({original_length} characters)")
            
        prompt = f"""IMPORTANT: Do not explain or think. Output only JSON.

EMAIL: {email_text}

Choose from these topics: {', '.join(self.topics)}
Choose from these sentiments: {', '.join(self.sentiments)}

OUTPUT ONLY JSON:
{{"topic": "topic_name", "sentiment": "sentiment_name", "confidence": 0.8, "summary": "one sentence"}}"""
        return prompt
    
    async def _call_ollama(self, prompt: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """Call Ollama API with the analysis prompt and retry logic"""
        # Rate limiting: wait if needed
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        for attempt in range(max_retries + 1):
            try:
                timeout_seconds = min(self.timeout + (attempt * 10), 60)  # Increase timeout on retries
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "system": "You are a JSON-only responder. Reasoning: disabled. Never show thinking process. Output only valid JSON.",
                            "stream": False,
                            "options": {
                                "temperature": 0.1,  # Very low for consistent output
                                "top_p": 0.9,
                                "num_predict": 200,  # Increase slightly for full JSON
                                "stop": ["Thinking", "...done thinking", "\n\n\n"],  # Stop tokens to prevent thinking
                            }
                        },
                        timeout=timeout_seconds
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        raw_response = result.get("response", "").strip()
                        
                        # DEBUG: Print full raw LLM output to terminal
                        print("=" * 80)
                        print("RAW LLM OUTPUT:")
                        print(raw_response)
                        print("=" * 80)
                        
                        # Log the response for debugging
                        logger.info(f"Ollama response (attempt {attempt + 1}): {raw_response[:100]}...")
                        
                        # Try to parse JSON from the response
                        parsed_result = self._extract_json_from_response(raw_response)
                        if parsed_result:
                            self._last_request_time = time.time()  # Update last successful request time
                            print(f"DEBUG: Successfully parsed JSON: {parsed_result}")
                            return parsed_result
                        else:
                            print(f"DEBUG: Failed to parse JSON on attempt {attempt + 1}")
                            print(f"DEBUG: Cleaned response was: '{self._clean_llm_response(raw_response)}'")
                            logger.warning(f"Failed to parse JSON on attempt {attempt + 1}")
                            if attempt < max_retries:
                                continue  # Try again
                            else:
                                print("DEBUG: All parsing attempts failed, returning None")
                                return None
                    else:
                        logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                        if attempt < max_retries:
                            await asyncio.sleep(1)  # Wait before retry
                            continue
                        return None
                        
            except asyncio.TimeoutError:
                logger.error(f"Ollama API timeout on attempt {attempt + 1} (timeout: {timeout_seconds}s)")
                if attempt < max_retries:
                    await asyncio.sleep(2)  # Wait before retry
                    continue
                return None
            except Exception as e:
                logger.error(f"Ollama API call failed on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(1)  # Wait before retry
                    continue
                return None
        
        return None
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response with improved handling"""
        try:
            # Log the raw response for debugging
            logger.info(f"Raw LLM response: {response[:200]}...")
            
            # Clean the response - remove thinking sections and extra text
            cleaned_response = self._clean_llm_response(response)
            
            # Try to parse the cleaned response directly as JSON
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                return json.loads(cleaned_response)
            
            # Use regex to find JSON objects
            import re
            json_pattern = r'\{[^{}]*"topic"[^{}]*"sentiment"[^{}]*"confidence"[^{}]*"summary"[^{}]*\}'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            
            if json_matches:
                # Try each match until we find valid JSON
                for match in json_matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # Fallback: look for any JSON-like structure
            json_pattern_loose = r'\{[^{}]*\}'
            json_matches_loose = re.findall(json_pattern_loose, response)
            
            for match in json_matches_loose:
                try:
                    parsed = json.loads(match)
                    # Check if it has our required fields
                    if all(key in parsed for key in ['topic', 'sentiment', 'confidence', 'summary']):
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            logger.warning(f"No valid JSON found in response: {response[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            logger.error(f"Response was: {response[:200]}...")
            return None
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response to extract just the JSON"""
        # Remove thinking sections - handle various formats
        thinking_patterns = [
            "Thinking...",
            "...done thinking.",
            "Let me think about this",
            "The user says:",
            "It seems they want",
            "Thus output:",
            "We need to ensure",
            "Check formatting:",
            "Probably fine.",
            "We'll output"
        ]
        
        for pattern in thinking_patterns:
            if pattern in response:
                # Split and take everything after the thinking section
                parts = response.split(pattern)
                if len(parts) > 1:
                    response = parts[-1]
        
        # Remove common prefixes/suffixes and clean up
        response = response.strip()
        
        # Remove any text before the first {
        start_idx = response.find('{')
        if start_idx >= 0:
            response = response[start_idx:]
        else:
            # No JSON found, return empty
            return ""
        
        # Remove any text after the last }
        end_idx = response.rfind('}')
        if end_idx != -1 and end_idx < len(response) - 1:
            response = response[:end_idx + 1]
        
        return response.strip()
    
    def _validate_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the analysis result"""
        try:
            # Validate required fields
            topic = result.get("topic", "other")
            sentiment = result.get("sentiment", "neutral")
            confidence = result.get("confidence", 0.5)
            summary = result.get("summary", "Email analysis completed")
            
            # Validate topic is in allowed list
            if topic not in self.topics:
                logger.warning(f"Invalid topic '{topic}', defaulting to 'other'")
                topic = "other"
            
            # Validate sentiment is in allowed list
            if sentiment not in self.sentiments:
                logger.warning(f"Invalid sentiment '{sentiment}', defaulting to 'neutral'")
                sentiment = "neutral"
            
            # Validate confidence is a number between 0 and 1
            try:
                confidence = float(confidence)
                confidence = max(0.1, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5
            
            # Ensure summary is a string and not too long
            summary = str(summary)[:200]
            
            return {
                "topic": topic,
                "sentiment": sentiment,
                "confidence": confidence,
                "summary": summary,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Result validation failed: {str(e)}")
            return self._create_fallback_result("Validation error")
    
    def _create_fallback_result(self, error_msg: str) -> Dict[str, Any]:
        """Create fallback result when LLM analysis fails"""
        return {
            "topic": "other",
            "sentiment": "neutral",
            "confidence": 0.1,
            "summary": f"Analysis failed: {error_msg}",
            "status": "fallback"
        }
    
    async def test_connection(self) -> bool:
        """Test if Ollama server and model are accessible"""
        try:
            async with httpx.AsyncClient() as client:
                # Test server connection
                response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                if response.status_code != 200:
                    return False
                
                # Check if our model is available
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                return self.model in model_names
                
        except Exception as e:
            logger.error(f"Ollama connection test failed: {str(e)}")
            return False


# Global LLM processor instance
llm_processor = LLMProcessor()
