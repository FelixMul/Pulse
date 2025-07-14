"""
Topic classifier training script for Parliament Pulse.
Fine-tunes DistilBERT for multi-label topic classification on constituent emails.
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, multilabel_confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

from data_preprocessing import EmailDataProcessor


class EmailTopicDataset(Dataset):
    """Dataset class for email topic classification."""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of email texts
            labels: Multi-label binary matrix
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
        labels = torch.FloatTensor(self.labels[idx])
        
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
            'labels': labels
        }


class TopicClassifierTrainer:
    """Handles training of the topic classification model."""
    
    def __init__(self, num_topics: int, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize the trainer.
        
        Args:
            num_topics: Number of topic classes
            model_name: HuggingFace model name
        """
        self.num_topics = num_topics
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
    def setup_model(self):
        """Initialize the DistilBERT model for multi-label classification."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_topics,
            problem_type="multi_label_classification"
        )
        
    def create_datasets(self, X_train: List[str], y_train: np.ndarray,
                       X_val: List[str], y_val: np.ndarray,
                       max_length: int = 512) -> Tuple[EmailTopicDataset, EmailTopicDataset]:
        """Create training and validation datasets."""
        train_dataset = EmailTopicDataset(X_train, y_train, self.tokenizer, max_length)
        val_dataset = EmailTopicDataset(X_val, y_val, self.tokenizer, max_length)
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics including cosine similarity."""
        predictions, labels = eval_pred
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(torch.from_numpy(predictions)).numpy()
        
        # For binary predictions (threshold = 0.5)
        predictions_binary = (predictions > 0.5).astype(int)
        
        # Compute metrics
        f1_micro = f1_score(labels, predictions_binary, average='micro')
        f1_macro = f1_score(labels, predictions_binary, average='macro')
        f1_weighted = f1_score(labels, predictions_binary, average='weighted')
        
        # Compute cosine similarity
        cosine_sim_scores = []
        for i in range(len(labels)):
            if labels[i].sum() > 0 and predictions[i].sum() > 0:  # Avoid division by zero
                cos_sim = cosine_similarity(
                    labels[i].reshape(1, -1),
                    predictions[i].reshape(1, -1)
                )[0, 0]
                cosine_sim_scores.append(cos_sim)
        
        avg_cosine_similarity = np.mean(cosine_sim_scores) if cosine_sim_scores else 0.0
        
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'cosine_similarity': avg_cosine_similarity
        }
    
    def train(self, train_dataset: EmailTopicDataset, val_dataset: EmailTopicDataset,
              output_dir: str = './models/topic_model',
              num_epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100):
        """Train the topic classification model."""
        
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
            metric_for_best_model="cosine_similarity",
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
    
    def evaluate_model(self, test_dataset: EmailTopicDataset, topic_names: List[str],
                      output_dir: str = './outputs/metrics') -> Dict:
        """Evaluate the trained model on test set."""
        print("Evaluating model on test set...")
        
        # Make predictions
        predictions = self.trainer.predict(test_dataset)
        logits = predictions.predictions
        labels = predictions.label_ids
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        pred_binary = (probs > 0.5).astype(int)
        
        # Compute comprehensive metrics
        metrics = self.compute_metrics((logits, labels))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, pred_binary, average=None, zero_division=0
        )
        
        # Create detailed results
        results = {
            'overall_metrics': metrics,
            'per_class_metrics': {
                topic_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(topic_names))
            }
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/topic_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self._create_evaluation_plots(labels, pred_binary, topic_names, output_dir)
        
        return results
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                                topic_names: List[str], output_dir: str):
        """Create evaluation plots."""
        # Confusion matrix for each topic
        cm_matrices = multilabel_confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (cm, topic) in enumerate(zip(cm_matrices[:6], topic_names[:6])):
            if i < len(axes):
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
                axes[i].set_title(f'{topic} Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topic_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Topic distribution plot
        topic_counts_true = y_true.sum(axis=0)
        topic_counts_pred = y_pred.sum(axis=0)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(topic_names))
        width = 0.35
        
        plt.bar(x - width/2, topic_counts_true, width, label='True', alpha=0.8)
        plt.bar(x + width/2, topic_counts_pred, width, label='Predicted', alpha=0.8)
        
        plt.xlabel('Topics')
        plt.ylabel('Count')
        plt.title('Topic Distribution: True vs Predicted')
        plt.xticks(x, topic_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train topic classification model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./models/topic_model', help='Output directory')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    processor = EmailDataProcessor()
    df = processor.load_data()
    
    # Prepare topic data
    texts, labels, topic_names = processor.prepare_topic_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(texts, labels)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of topics: {len(topic_names)}")
    print(f"Topics: {topic_names}")
    
    # Initialize trainer
    trainer = TopicClassifierTrainer(num_topics=len(topic_names))
    trainer.setup_model()
    
    # Create datasets
    train_dataset, val_dataset = trainer.create_datasets(
        X_train, y_train, X_val, y_val, args.max_length
    )
    test_dataset = EmailTopicDataset(X_test, y_test, trainer.tokenizer, args.max_length)
    
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
    results = trainer.evaluate_model(test_dataset, topic_names)
    
    print("\nEvaluation Results:")
    print(f"Overall F1 (micro): {results['overall_metrics']['f1_micro']:.4f}")
    print(f"Overall F1 (macro): {results['overall_metrics']['f1_macro']:.4f}")
    print(f"Average Cosine Similarity: {results['overall_metrics']['cosine_similarity']:.4f}")
    
    print("\nPer-topic F1 scores:")
    for topic, metrics in results['per_class_metrics'].items():
        print(f"{topic}: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main() 