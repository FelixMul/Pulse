"""
Model evaluation script for Parliament Pulse.
Evaluates trained topic and sentiment classification models.
"""

import argparse
import json
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from data_preprocessing import EmailDataProcessor
from topic_classifier import EmailTopicDataset
from sentiment_classifier import EmailSentimentDataset


class ModelEvaluator:
    """Handles evaluation of trained models."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_topic_model(self, model_path: str):
        """Load a trained topic classification model."""
        print(f"Loading topic model from {model_path}")
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def load_sentiment_model(self, model_path: str):
        """Load a trained sentiment classification model."""
        print(f"Loading sentiment model from {model_path}")
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def predict_topics(self, model, tokenizer, texts: List[str], threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Make topic predictions on a list of texts."""
        predictions = []
        probabilities = []
        
        model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Apply sigmoid for multi-label
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                pred = (probs > threshold).astype(int)
                
                predictions.append(pred)
                probabilities.append(probs)
        
        return np.array(predictions), np.array(probabilities)
    
    def predict_sentiment(self, model, tokenizer, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make sentiment predictions on a list of texts."""
        predictions = []
        probabilities = []
        
        model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Apply softmax for multi-class
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = np.argmax(probs)
                
                predictions.append(pred)
                probabilities.append(probs)
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate_topic_model(self, model, tokenizer, test_texts: List[str], 
                           test_labels: np.ndarray, topic_names: List[str]) -> Dict:
        """Evaluate topic classification model."""
        print("Evaluating topic model...")
        
        # Make predictions
        pred_labels, pred_probs = self.predict_topics(model, tokenizer, test_texts)
        
        # Compute metrics
        f1_micro = f1_score(test_labels, pred_labels, average='micro')
        f1_macro = f1_score(test_labels, pred_labels, average='macro')
        f1_weighted = f1_score(test_labels, pred_labels, average='weighted')
        
        # Compute cosine similarity
        cosine_sim_scores = []
        for i in range(len(test_labels)):
            if test_labels[i].sum() > 0 and pred_probs[i].sum() > 0:
                cos_sim = cosine_similarity(
                    test_labels[i].reshape(1, -1),
                    pred_probs[i].reshape(1, -1)
                )[0, 0]
                cosine_sim_scores.append(cos_sim)
        
        avg_cosine_similarity = np.mean(cosine_sim_scores) if cosine_sim_scores else 0.0
        
        # Per-class metrics
        per_class_f1 = f1_score(test_labels, pred_labels, average=None)
        
        results = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'cosine_similarity': avg_cosine_similarity,
            'per_class_f1': {topic_names[i]: float(per_class_f1[i]) for i in range(len(topic_names))}
        }
        
        return results
    
    def evaluate_sentiment_model(self, model, tokenizer, test_texts: List[str], 
                                test_labels: np.ndarray) -> Dict:
        """Evaluate sentiment classification model."""
        print("Evaluating sentiment model...")
        
        # Make predictions
        pred_labels, pred_probs = self.predict_sentiment(model, tokenizer, test_texts)
        
        # Compute metrics
        accuracy = accuracy_score(test_labels, pred_labels)
        f1_micro = f1_score(test_labels, pred_labels, average='micro')
        f1_macro = f1_score(test_labels, pred_labels, average='macro')
        f1_weighted = f1_score(test_labels, pred_labels, average='weighted')
        
        # Per-class metrics
        per_class_f1 = f1_score(test_labels, pred_labels, average=None)
        sentiment_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive', 'Mixed']
        
        results = {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_f1': {sentiment_names[i]: float(per_class_f1[i]) for i in range(len(sentiment_names))}
        }
        
        return results
    
    def evaluate_sample_predictions(self, model, tokenizer, texts: List[str], 
                                  model_type: str = 'topic', n_samples: int = 5):
        """Show sample predictions for manual inspection."""
        print(f"\nSample {model_type} predictions:")
        print("=" * 50)
        
        if model_type == 'topic':
            predictions, probabilities = self.predict_topics(model, tokenizer, texts[:n_samples])
            topic_names = ['Healthcare', 'Economy', 'Education', 'Environment', 'Immigration', 
                          'Infrastructure', 'Social Issues', 'Security', 'Local Matters', 'Other']
            
            for i, (text, pred, probs) in enumerate(zip(texts[:n_samples], predictions, probabilities)):
                print(f"\nExample {i+1}:")
                print(f"Text: {text[:100]}...")
                predicted_topics = [topic_names[j] for j, p in enumerate(pred) if p == 1]
                print(f"Predicted topics: {predicted_topics}")
                top_probs = sorted(zip(topic_names, probs), key=lambda x: x[1], reverse=True)[:3]
                print(f"Top probabilities: {[(t, f'{p:.3f}') for t, p in top_probs]}")
        
        else:  # sentiment
            predictions, probabilities = self.predict_sentiment(model, tokenizer, texts[:n_samples])
            sentiment_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive', 'Mixed']
            
            for i, (text, pred, probs) in enumerate(zip(texts[:n_samples], predictions, probabilities)):
                print(f"\nExample {i+1}:")
                print(f"Text: {text[:100]}...")
                print(f"Predicted sentiment: {sentiment_names[pred]}")
                top_probs = sorted(zip(sentiment_names, probs), key=lambda x: x[1], reverse=True)[:3]
                print(f"Top probabilities: {[(s, f'{p:.3f}') for s, p in top_probs]}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_type', type=str, choices=['topic', 'sentiment', 'both'], 
                       default='both', help='Which model(s) to evaluate')
    parser.add_argument('--topic_model_path', type=str, default='./models/topic_model',
                       help='Path to trained topic model')
    parser.add_argument('--sentiment_model_path', type=str, default='./models/sentiment_model',
                       help='Path to trained sentiment model')
    parser.add_argument('--output_dir', type=str, default='./outputs/metrics',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load data
    processor = EmailDataProcessor()
    df = processor.load_data()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    if args.model_type in ['topic', 'both']:
        try:
            # Prepare topic data
            texts, labels, topic_names = processor.prepare_topic_data(df)
            _, _, X_test, _, _, y_test = processor.split_data(texts, labels)
            
            # Load and evaluate topic model
            topic_model, topic_tokenizer = evaluator.load_topic_model(args.topic_model_path)
            topic_results = evaluator.evaluate_topic_model(
                topic_model, topic_tokenizer, X_test, y_test, topic_names
            )
            
            print("\nTopic Model Results:")
            print(f"F1 (micro): {topic_results['f1_micro']:.4f}")
            print(f"F1 (macro): {topic_results['f1_macro']:.4f}")
            print(f"F1 (weighted): {topic_results['f1_weighted']:.4f}")
            print(f"Cosine Similarity: {topic_results['cosine_similarity']:.4f}")
            
            # Show sample predictions
            evaluator.evaluate_sample_predictions(topic_model, topic_tokenizer, X_test, 'topic')
            
            # Save results
            with open(f'{args.output_dir}/topic_evaluation.json', 'w') as f:
                json.dump(topic_results, f, indent=2)
                
        except Exception as e:
            print(f"Error evaluating topic model: {e}")
    
    if args.model_type in ['sentiment', 'both']:
        try:
            # Prepare sentiment data
            texts, labels = processor.prepare_sentiment_data(df)
            _, _, X_test, _, _, y_test = processor.split_data(texts, labels)
            
            # Load and evaluate sentiment model
            sentiment_model, sentiment_tokenizer = evaluator.load_sentiment_model(args.sentiment_model_path)
            sentiment_results = evaluator.evaluate_sentiment_model(
                sentiment_model, sentiment_tokenizer, X_test, y_test
            )
            
            print("\nSentiment Model Results:")
            print(f"Accuracy: {sentiment_results['accuracy']:.4f}")
            print(f"F1 (macro): {sentiment_results['f1_macro']:.4f}")
            print(f"F1 (weighted): {sentiment_results['f1_weighted']:.4f}")
            
            # Show sample predictions
            evaluator.evaluate_sample_predictions(sentiment_model, sentiment_tokenizer, X_test, 'sentiment')
            
            # Save results
            with open(f'{args.output_dir}/sentiment_evaluation.json', 'w') as f:
                json.dump(sentiment_results, f, indent=2)
                
        except Exception as e:
            print(f"Error evaluating sentiment model: {e}")


if __name__ == "__main__":
    main() 