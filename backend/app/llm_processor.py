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
    
    async def analyze_email(self, email_text: str, model_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze email using local LLM for topic and sentiment
        Returns structured JSON with topic, sentiment, confidence, and summary
        """
        try:
            # Create structured prompt
            prompt = self._create_analysis_prompt(email_text)
            
            # Call Ollama API
            analysis_result = await self._call_ollama(prompt, model_tag=model_tag)
            
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
            
        # Instruct model to include a single fenced JSON block we can extract reliably
        prompt = (
            "You are an assistant that ALWAYS includes exactly one fenced JSON code block in your reply.\n"
            "You may include brief text before/after, but MUST include a Markdown JSON block like this:\n\n"
            "```json\n"
            "{\n"
            "  \"topic\": \"...\",\n"
            "  \"sentiment\": \"very_negative|negative|neutral|positive|very_positive\",\n"
            "  \"confidence\": 0.0,\n"
            "  \"summary\": \"one sentence\"\n"
            "}\n"
            "```\n\n"
            "Rules:\n"
            "- The JSON must be syntactically valid and parseable.\n"
            "- Use one of these topics exactly: " + ", ".join(self.topics) + "\n"
            "- Use one of these sentiments exactly: " + ", ".join(self.sentiments) + "\n"
            "- confidence is a number between 0.0 and 1.0.\n\n"
            f"EMAIL: {email_text}\n"
        )
        return prompt
    
    async def _call_ollama(self, prompt: str, model_tag: Optional[str] = None, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """Call Ollama API with the analysis prompt and retry logic"""
        # Rate limiting: wait if needed
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        chosen_model = (model_tag or self.model)
        
        for attempt in range(max_retries + 1):
            try:
                timeout_seconds = min(self.timeout + (attempt * 10), 60)  # Increase timeout on retries
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": chosen_model,
                            "prompt": prompt,
                            "system": "You are a JSON-only responder. Reasoning: disabled. Never show thinking process. Output only valid JSON.",
                            "stream": False,
                            "options": {
                                "temperature": 0.1,  # deterministic
                                "top_p": 0.95,
                                "num_predict": 1024,  # allow enough tokens for full JSON
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
            
            # 1) Prefer fenced JSON blocks: ```json ... ``` or ``` ... ```
            import re
            fenced_json = re.search(r"```json\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
            if fenced_json:
                candidate = fenced_json.group(1).strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

            fenced_any = re.search(r"```\s*([\s\S]*?)\s*```", response)
            if fenced_any:
                candidate = fenced_any.group(1).strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

            # 2) Clean the response of boilerplate and try direct parse
            cleaned_response = self._clean_llm_response(response)
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                try:
                    return json.loads(cleaned_response)
                except json.JSONDecodeError:
                    pass
            
            # 3) Use regex to find probable objects with required keys
            json_pattern = r'\{[\s\S]*?\}'
            json_matches = re.findall(json_pattern, response)
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if all(k in parsed for k in ["topic", "sentiment", "confidence", "summary"]):
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            # 4) Attempt balanced-brace extraction for the largest object
            balanced = self._extract_first_balanced_json(response)
            if balanced:
                try:
                    parsed = json.loads(balanced)
                    if all(k in parsed for k in ["topic", "sentiment", "confidence", "summary"]):
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"No valid JSON found in response: {response[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            logger.error(f"Response was: {response[:200]}...")
            return None

    def _extract_first_balanced_json(self, text: str) -> Optional[str]:
        """Scan text to extract the first balanced {...} block."""
        start = -1
        depth = 0
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return text[start:i+1]
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
    
    async def test_connection(self, model_tag: Optional[str] = None) -> bool:
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
                check_model = (model_tag or self.model)
                return check_model in model_names
                
        except Exception as e:
            logger.error(f"Ollama connection test failed: {str(e)}")
            return False


# Global LLM processor instance
llm_processor = LLMProcessor()
