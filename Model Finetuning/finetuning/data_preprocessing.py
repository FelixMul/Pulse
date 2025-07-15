"""
Data preprocessing for Parliament Pulse email classification models.
Loads and prepares the synthetic email dataset for training DistilBERT models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
try:
    from skmultilearn.model_selection import iterative_train_test_split
    ITERATIVE_SPLIT_AVAILABLE = True
except ImportError:
    ITERATIVE_SPLIT_AVAILABLE = False
import torch
from transformers import DistilBertTokenizer

class EmailDataProcessor:
    """Handles loading and preprocessing of the email dataset for model training."""
    
    def __init__(self, data_path: str = "../data_creation/emails_data/email_dataset_final.csv"):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the email dataset CSV file
        """
        self.data_path = data_path
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.mlb = MultiLabelBinarizer()
        self.sentiment_mapping = {
            'Very Negative': 0,
            'Negative': 1, 
            'Neutral': 2,
            'Positive': 3,
            'Very Positive': 4,
            'Mixed': 5
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load the email dataset from CSV."""
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} emails")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def parse_topics(self, topics_str: str) -> List[str]:
        """
        Parse topics from string representation to list.
        
        Args:
            topics_str: String representation of topics (e.g., "['Healthcare', 'Economy']")
            
        Returns:
            List of topic strings
        """
        try:
            # Handle string representation of lists
            if isinstance(topics_str, str):
                # Remove extra quotes and parse
                topics_str = topics_str.strip()
                if topics_str.startswith("'") and topics_str.endswith("'"):
                    topics_str = topics_str[1:-1]
                topics = ast.literal_eval(topics_str)
            else:
                topics = topics_str
                
            # Ensure it's a list
            if isinstance(topics, str):
                topics = [topics]
            elif not isinstance(topics, list):
                topics = [str(topics)]
                
            return topics
        except:
            # Fallback for malformed data
            print(f"Warning: Could not parse topics: {topics_str}")
            return ["Other"]
    
    def clean_email_text(self, text: str) -> str:
        """
        Clean email text for model input.
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned email text
        """
        if pd.isna(text):
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove email headers/footers patterns (basic)
        text = re.sub(r'From:.*?Subject:', '', text, flags=re.DOTALL)
        text = re.sub(r'Best regards.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Yours sincerely.*$', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def prepare_topic_data(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Prepare data for topic classification (multi-label).
        
        Args:
            df: DataFrame with email data
            
        Returns:
            Tuple of (texts, topic_labels, unique_topics)
        """
        print("Preparing topic classification data...")
        
        # Parse topics
        topics_lists = df['topics'].apply(self.parse_topics)
        
        # Get all unique topics
        all_topics = set()
        for topics in topics_lists:
            all_topics.update(topics)
        unique_topics = sorted(list(all_topics))
        print(f"Found {len(unique_topics)} unique topics: {unique_topics}")
        
        # Convert to multi-label binary format
        self.mlb.fit([unique_topics])
        topic_labels = self.mlb.transform(topics_lists)
        
        # Clean email texts
        texts = df['email_body'].apply(self.clean_email_text).tolist()
        
        print(f"Topic data shape: {topic_labels.shape}")
        print(f"Sample topic distribution: {topic_labels.mean(axis=0)}")
        
        return texts, topic_labels, unique_topics
    
    def prepare_sentiment_data(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """
        Prepare data for sentiment classification (single-label).
        
        Args:
            df: DataFrame with email data
            
        Returns:
            Tuple of (texts, sentiment_labels)
        """
        print("Preparing sentiment classification data...")
        
        # Map sentiment strings to integers
        sentiment_labels = df['sentiment'].map(self.sentiment_mapping)
        
        # Check for unmapped sentiments
        if sentiment_labels.isna().any():
            unmapped = df.loc[sentiment_labels.isna(), 'sentiment'].unique()
            print(f"Warning: Unmapped sentiments found: {unmapped}")
            # Fill with neutral for unmapped
            sentiment_labels = sentiment_labels.fillna(2)
        
        # Clean email texts
        texts = df['email_body'].apply(self.clean_email_text).tolist()
        
        sentiment_counts = df['sentiment'].value_counts()
        print(f"Sentiment distribution:\n{sentiment_counts}")
        
        return texts, sentiment_labels.values
    
    def tokenize_texts(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts using DistilBERT tokenizer.
        
        Args:
            texts: List of email texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs
        """
        print(f"Tokenizing {len(texts)} texts...")
        
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        print(f"Tokenized shape: {encoded['input_ids'].shape}")
        return encoded
    
    def split_data(self, texts: List[str], labels: np.ndarray, test_size: float = 0.2, 
                   val_size: float = 0.1, random_state: int = 42) -> Tuple:
        """
        Split data into train/validation/test sets.
        
        Args:
            texts: List of email texts
            labels: Label array
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Tuple of split data (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"Splitting data: train/val/test")
        
        # Check if multi-label (2D array) or single-label (1D array)
        is_multilabel = len(labels.shape) > 1 and labels.shape[1] > 1
        
        if is_multilabel:
            # For multi-label, try iterative stratification if available, otherwise random
            if ITERATIVE_SPLIT_AVAILABLE:
                print("Multi-label detected - using iterative stratification")
                try:
                    # Convert to format expected by iterative_train_test_split
                    X_array = np.array(texts).reshape(-1, 1)
                    X_temp, y_temp, X_test, y_test = iterative_train_test_split(
                        X_array, labels, test_size=test_size
                    )
                    X_temp = X_temp.flatten()
                    X_test = X_test.flatten()
                    
                    # Second split: train vs val
                    val_size_adjusted = val_size / (1 - test_size)
                    X_train, y_train, X_val, y_val = iterative_train_test_split(
                        X_temp.reshape(-1, 1), y_temp, test_size=val_size_adjusted
                    )
                    X_train = X_train.flatten()
                    X_val = X_val.flatten()
                except Exception as e:
                    print(f"Iterative stratification failed: {e}, falling back to random split")
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        texts, labels, test_size=test_size, random_state=random_state
                    )
                    val_size_adjusted = val_size / (1 - test_size)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
                    )
            else:
                print("Multi-label detected - using random split (skmultilearn not available)")
                X_temp, X_test, y_temp, y_test = train_test_split(
                    texts, labels, test_size=test_size, random_state=random_state
                )
                
                # Second split: train vs val
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
                )
        else:
            # Single-label: can use stratify
            print("Single-label detected - using stratified split")
            X_temp, X_test, y_temp, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
            )
        
        print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame with email data
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'total_emails': len(df),
            'columns': list(df.columns),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_email_length': df['email_body'].str.len().mean(),
            'personas': df['persona'].nunique() if 'persona' in df.columns else 'N/A',
            'length_categories': df['length'].value_counts().to_dict() if 'length' in df.columns else 'N/A'
        }
        
        # Topic information
        if 'topics' in df.columns:
            topics_lists = df['topics'].apply(self.parse_topics)
            all_topics = []
            for topics in topics_lists:
                all_topics.extend(topics)
            topic_counts = pd.Series(all_topics).value_counts()
            info['topic_distribution'] = topic_counts.to_dict()
            info['unique_topics'] = len(topic_counts)
            info['avg_topics_per_email'] = len(all_topics) / len(df)
        
        return info


def main():
    """Example usage of the EmailDataProcessor."""
    processor = EmailDataProcessor()
    
    # Load data
    df = processor.load_data()
    
    # Get dataset info
    info = processor.get_dataset_info(df)
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Prepare topic data
    topic_texts, topic_labels, unique_topics = processor.prepare_topic_data(df)
    print(f"\nTopic model ready: {len(topic_texts)} texts, {topic_labels.shape[1]} topics")
    
    # Prepare sentiment data
    sentiment_texts, sentiment_labels = processor.prepare_sentiment_data(df)
    print(f"Sentiment model ready: {len(sentiment_texts)} texts, {len(np.unique(sentiment_labels))} classes")


if __name__ == "__main__":
    main() 