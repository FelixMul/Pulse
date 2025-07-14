"""
Sentiment classifier training script for Parliament Pulse.
Fine-tunes DistilBERT for sentiment classification on constituent emails.
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from data_preprocessing import EmailDataProcessor


class EmailSentimentDataset(Dataset):
    """Dataset class for email sentiment classification."""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of email texts
            labels: Sentiment labels (integers)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label.flatten()
        }


class SentimentClassifierTrainer:
    """Handles training of the sentiment classification model."""
    
    def __init__(self, num_labels: int = 6, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize the trainer.
        
        Args:
            num_labels: Number of sentiment classes
            model_name: HuggingFace model name
        """
        self.num_labels = num_labels
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        self.label_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive', 'Mixed']
        
    def setup_model(self):
        """Initialize the DistilBERT model for sentiment classification."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
    def create_datasets(self, X_train: List[str], y_train: np.ndarray,
                       X_val: List[str], y_val: np.ndarray,
                       max_length: int = 512) -> Tuple[EmailSentimentDataset, EmailSentimentDataset]:
        """Create training and validation datasets."""
        train_dataset = EmailSentimentDataset(X_train, y_train, self.tokenizer, max_length)
        val_dataset = EmailSentimentDataset(X_val, y_val, self.tokenizer, max_length)
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for sentiment classification."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def train(self, train_dataset: EmailSentimentDataset, val_dataset: EmailSentimentDataset,
              output_dir: str = './models/sentiment_model',
              num_epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100):
        """Train the sentiment classification model."""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            save_total_limit=2,
            push_to_hub=False,
            report_to=None  # Disable wandb logging by default
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("Starting training...")
        self.trainer.train()
        
        # Save the best model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    def evaluate_model(self, test_dataset: EmailSentimentDataset,
                      output_dir: str = './outputs/metrics') -> Dict:
        """Evaluate the trained model on test set."""
        print("Evaluating model on test set...")
        
        # Make predictions
        predictions = self.trainer.predict(test_dataset)
        logits = predictions.predictions
        labels = predictions.label_ids
        pred_labels = np.argmax(logits, axis=1)
        
        # Compute comprehensive metrics
        metrics = self.compute_metrics((logits, labels))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, pred_labels, average=None, zero_division=0
        )
        
        # Create detailed results
        results = {
            'overall_metrics': metrics,
            'per_class_metrics': {
                self.label_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(self.label_names))
            },
            'confusion_matrix': confusion_matrix(labels, pred_labels).tolist()
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/sentiment_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self._create_evaluation_plots(labels, pred_labels, output_dir)
        
        return results
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
        """Create evaluation plots."""
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_names,
                    yticklabels=self.label_names)
        plt.title('Sentiment Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Class distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        # Ensure all classes are represented
        all_counts_true = np.zeros(len(self.label_names))
        all_counts_pred = np.zeros(len(self.label_names))
        
        for i, count in zip(unique_true, counts_true):
            all_counts_true[i] = count
        for i, count in zip(unique_pred, counts_pred):
            all_counts_pred[i] = count
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.label_names))
        width = 0.35
        
        plt.bar(x - width/2, all_counts_true, width, label='True', alpha=0.8)
        plt.bar(x + width/2, all_counts_pred, width, label='Predicted', alpha=0.8)
        
        plt.xlabel('Sentiment Classes')
        plt.ylabel('Count')
        plt.title('Sentiment Distribution: True vs Predicted')
        plt.xticks(x, self.label_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train sentiment classification model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./models/sentiment_model', help='Output directory')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    processor = EmailDataProcessor()
    df = processor.load_data()
    
    # Prepare sentiment data
    texts, labels = processor.prepare_sentiment_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(texts, labels)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of sentiment classes: {len(np.unique(labels))}")
    
    # Initialize trainer
    trainer = SentimentClassifierTrainer(num_labels=len(np.unique(labels)))
    trainer.setup_model()
    
    # Create datasets
    train_dataset, val_dataset = trainer.create_datasets(
        X_train, y_train, X_val, y_val, args.max_length
    )
    test_dataset = EmailSentimentDataset(X_test, y_test, trainer.tokenizer, args.max_length)
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate model
    results = trainer.evaluate_model(test_dataset)
    
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
    print(f"Overall F1 (macro): {results['overall_metrics']['f1_macro']:.4f}")
    print(f"Overall F1 (weighted): {results['overall_metrics']['f1_weighted']:.4f}")
    
    print("\nPer-class F1 scores:")
    for sentiment, metrics in results['per_class_metrics'].items():
        print(f"{sentiment}: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main() 