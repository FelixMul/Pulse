"""
NLP Processing Pipeline for Parliament Pulse
Handles spam detection, topic modeling, and sentiment analysis
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pickle
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch

from .config import settings

logger = logging.getLogger(__name__)

class TextCleaner:
    """Text preprocessing and cleaning utilities"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        
        # Initialize lemmatizer
        try:
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            self.lemmatizer = WordNetLemmatizer()
            self.tokenizer = word_tokenize
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK lemmatizer: {str(e)}")
            self.lemmatizer = None
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses (but keep the fact that they exist)
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Remove URLs (but keep the fact that they exist)
        text = self.url_pattern.sub('[URL]', text)
        
        # Remove phone numbers (but keep the fact that they exist)
        text = self.phone_pattern.sub('[PHONE]', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\[\]]', ' ', text)
        
        # Apply lemmatization if available
        if self.lemmatizer:
            try:
                # Tokenize text
                tokens = self.tokenizer(text)
                
                # Lemmatize and remove stopwords
                lemmatized_tokens = []
                for token in tokens:
                    if token.isalpha() and token not in self.stop_words and len(token) > 2:
                        lemmatized_token = self.lemmatizer.lemmatize(token)
                        lemmatized_tokens.append(lemmatized_token)
                
                # Rejoin tokens
                text = ' '.join(lemmatized_tokens)
                
            except Exception as e:
                logger.warning(f"Lemmatization failed: {str(e)}")
        
        return text.strip()
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract additional features from text"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'has_email': bool(self.email_pattern.search(text)),
            'has_url': bool(self.url_pattern.search(text)),
            'has_phone': bool(self.phone_pattern.search(text)),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?')
        }
        return features

class SpamDetector:
    """Spam detection using Naive Bayes classifier"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.model_path = os.path.join(settings.DATA_DIR, 'spam_model.pkl')
        
    def train_on_sample_data(self):
        """Train spam detector on sample data (for initial setup)"""
        try:
            # Sample training data (in production, this would be replaced with real data)
            spam_samples = [
                "congratulations you have won 1000000 dollars click here now urgent",
                "free money now limited time offer click here urgent",
                "urgent reply needed send money inheritance millions",
                "weight loss miracle cure free trial limited time",
                "make money fast work from home opportunity",
                "free viagra cialis medical offer online pharmacy",
                "lottery winner claim prize money urgent response needed",
                "business proposal urgent confidential reply needed millions"
            ]
            
            legitimate_samples = [
                "thank you for your support on healthcare policy",
                "concerned about local transportation infrastructure development",
                "request meeting discuss education funding proposals",
                "neighborhood safety community policing initiative support",
                "environmental protection legislation feedback constituent",
                "job creation programs local economic development",
                "senior citizen services medicare support needed",
                "small business tax relief policy discussion request"
            ]
            
            # Create training dataset
            texts = spam_samples + legitimate_samples
            labels = ['spam'] * len(spam_samples) + ['legitimate'] * len(legitimate_samples)
            
            # Train the model
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
            
            self.is_trained = True
            self.save_model()
            
            logger.info("Spam detector trained on sample data")
            
        except Exception as e:
            logger.error(f"Failed to train spam detector: {str(e)}")
            
    def save_model(self):
        """Save trained model to disk"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Spam model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save spam model: {str(e)}")
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.is_trained = model_data['is_trained']
                
                logger.info("Spam model loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to load spam model: {str(e)}")
        return False
    
    def detect_spam(self, text: str) -> Dict[str, Any]:
        """Detect if text is spam"""
        try:
            if not self.is_trained:
                if not self.load_model():
                    self.train_on_sample_data()
            
            if not self.is_trained:
                return {'is_spam': False, 'confidence': 0.0, 'error': 'Model not trained'}
            
            # Vectorize the text
            X = self.vectorizer.transform([text])
            
            # Predict
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Get confidence score
            is_spam = prediction == 'spam'
            confidence = max(probabilities)
            
            return {
                'is_spam': is_spam,
                'confidence': float(confidence),
                'prediction': prediction
            }
            
        except Exception as e:
            logger.error(f"Spam detection failed: {str(e)}")
            return {'is_spam': False, 'confidence': 0.0, 'error': str(e)}

class SentimentAnalyzer:
    """Sentiment analysis using VADER"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            scores = self.analyzer.polarity_scores(text)
            
            # Determine overall sentiment
            compound = scores['compound']
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Return in the expected structure with nested scores
            return {
                'sentiment': sentiment,
                'scores': {
                    'compound': compound,
                    'positive': scores['pos'],
                    'neutral': scores['neu'],
                    'negative': scores['neg']
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                'sentiment': 'neutral',
                'scores': {
                    'compound': 0.0,
                    'positive': 0.0,
                    'neutral': 1.0,
                    'negative': 0.0
                },
                'error': str(e)
            }

class TopicClassifier:
    """Topic classification using zero-shot classification"""
    
    def __init__(self):
        self.classifier = None
        self.is_ready = False
        
        # Define predefined political topics
        self.candidate_topics = [
            "Healthcare", "Transportation", "Education", "Housing",
            "Local Business", "Environment", "Public Safety", "Economy",
            "Immigration", "Taxes", "Social Services", "Infrastructure",
            "Climate Change", "Employment", "Senior Services", "Youth Programs"
        ]
        
        # Load classifier immediately at startup
        logger.info("Initializing topic classifier at startup...")
        self._load_classifier()
    
    def _load_classifier(self) -> bool:
        """Load the zero-shot classification model"""
        try:
            if self.classifier is None:
                logger.info("Loading zero-shot classification model...")
                self.classifier = pipeline(
                    "zero-shot-classification", 
                    model="typeform/distilbert-base-uncased-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.is_ready = True
                logger.info("Zero-shot classifier loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load zero-shot classifier: {str(e)}")
            return False

    def classify_topic(self, text: str) -> Dict[str, Any]:
        """Classify topic using zero-shot classification"""
        try:
            # Check if classifier is ready (should be loaded at startup)
            if not self.is_ready or self.classifier is None:
                return {
                    'topic_id': -1, 
                    'topic_label': 'Classifier Not Ready', 
                    'keywords': [], 
                    'confidence': 0.0,
                    'error': 'Classifier was not properly initialized at startup'
                }
            
            # Run zero-shot classification
            results = self.classifier(text, self.candidate_topics, multi_label=True)
            
            # Get the top topic
            if results['scores'] and len(results['scores']) > 0:
                top_topic = results['labels'][0]
                top_confidence = results['scores'][0]
                
                # Get additional context (top 3 topics)
                top_topics = results['labels'][:3]
                top_scores = results['scores'][:3]
                
                return {
                    'topic_id': self.candidate_topics.index(top_topic),
                    'topic_label': top_topic,
                    'keywords': top_topics,  # Show top 3 related topics
                    'confidence': float(top_confidence),
                    'all_scores': dict(zip(top_topics, [float(s) for s in top_scores]))
                }
            else:
                return {
                    'topic_id': -1,
                    'topic_label': 'Unknown',
                    'keywords': [],
                    'confidence': 0.0
                }
            
        except Exception as e:
            logger.error(f"Topic classification failed: {str(e)}")
            return {
                'topic_id': -1, 
                'topic_label': 'Error', 
                'keywords': [], 
                'confidence': 0.0, 
                'error': str(e)
            }
    
    def get_available_topics(self) -> List[Dict[str, Any]]:
        """Get list of available topic categories"""
        try:
            topics = []
            for i, topic in enumerate(self.candidate_topics):
                topics.append({
                    'topic_id': i,
                    'topic_label': topic,
                    'description': f"Issues related to {topic.lower()}"
                })
            return topics
        except Exception as e:
            logger.error(f"Failed to get available topics: {str(e)}")
            return []

class NLPProcessor:
    """Main NLP processing pipeline"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.spam_detector = SpamDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_classifier = TopicClassifier()
        
        # Download required NLTK data
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Download required NLTK datasets
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('wordnet', quiet=True)  # For lemmatization
            nltk.download('punkt_tab', quiet=True)  # For improved tokenization
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {str(e)}")
    
    def process_email(self, email_text: str, email_id: str = None) -> Dict[str, Any]:
        """Process a single email through the full NLP pipeline"""
        try:
            # Clean the text
            cleaned_text = self.text_cleaner.clean_text(email_text)
            
            # Extract text features
            text_features = self.text_cleaner.extract_features(email_text)
            
            # Spam detection
            spam_result = self.spam_detector.detect_spam(cleaned_text)
            
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(cleaned_text)
            
            # Topic classification
            topic_result = self.topic_classifier.classify_topic(cleaned_text)
            
            return {
                'email_id': email_id,
                'cleaned_text': cleaned_text,
                'text_features': text_features,
                'spam_detection': spam_result,
                'sentiment_analysis': sentiment_result,
                'topic_modeling': topic_result,
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"NLP processing failed for email {email_id}: {str(e)}")
            return {
                'email_id': email_id,
                'error': str(e),
                'processed_at': datetime.utcnow().isoformat()
            }
    
    def process_batch(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of emails"""
        results = []
        
        for email in emails:
            email_text = email.get('raw_body', '') or email.get('body', '')
            email_id = email.get('id')
            
            result = self.process_email(email_text, email_id)
            results.append(result)
        
        return results
    

    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all NLP models for dashboard monitoring"""
        try:
            status = {
                'sentiment_analyzer': {
                    'ready': self.sentiment_analyzer is not None,
                    'model_type': 'VADER',
                    'last_updated': None
                },
                'spam_detector': {
                    'ready': self.spam_detector is not None,
                    'trained': False,
                    'model_type': 'Naive Bayes',
                    'last_updated': None
                },
                'topic_classifier': {
                    'ready': self.topic_classifier.is_ready,
                    'model_type': 'Zero-Shot Classification (DistilBERT)',
                    'model_name': 'typeform/distilbert-base-uncased-mnli',
                    'available_topics': len(self.topic_classifier.candidate_topics),
                    'last_updated': None
                },
                'text_cleaner': {
                    'ready': self.text_cleaner is not None,
                    'last_updated': None
                }
            }
            
            # Check if spam detector has a trained model
            if self.spam_detector and hasattr(self.spam_detector, 'model') and self.spam_detector.model:
                status['spam_detector']['trained'] = True
                # Check if model file exists
                if os.path.exists(self.spam_detector.model_path):
                    model_stat = os.stat(self.spam_detector.model_path)
                    status['spam_detector']['last_updated'] = datetime.fromtimestamp(model_stat.st_mtime).isoformat()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status: {str(e)}")
            return {
                'sentiment_analyzer': {'ready': False, 'error': str(e)},
                'spam_detector': {'ready': False, 'trained': False, 'error': str(e)},
                'topic_classifier': {'ready': False, 'model_name': 'typeform/distilbert-base-uncased-mnli', 'error': str(e)},
                'text_cleaner': {'ready': False, 'error': str(e)}
            }

# Note: NLP processor is now initialized at FastAPI startup
# No more lazy loading - models load when the server starts 