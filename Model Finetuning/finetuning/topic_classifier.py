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
from transformers.integrations import TensorBoardCallback

from data_preprocessing import EmailDataProcessor


def download_and_cache_tokenizer(model_name: str = 'distilbert-base-uncased', cache_dir: str = './cached_models'):
    """
    Download and cache HuggingFace tokenizer locally.
    Note: We only cache the tokenizer, not the model architecture, to avoid num_labels conflicts.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Local directory to cache tokenizer
    
    Returns:
        Tuple of (model_name, tokenizer_path)
    """
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_cache_path = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_tokenizer")
    
    # Check if tokenizer is already cached
    if os.path.exists(tokenizer_cache_path) and os.listdir(tokenizer_cache_path):
        print(f"Using cached tokenizer from {tokenizer_cache_path}")
        return model_name, tokenizer_cache_path
    
    print(f"Downloading and caching tokenizer for {model_name} to {tokenizer_cache_path}")
    
    # Download and save tokenizer only
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_cache_path)
    
    print(f"Tokenizer cached successfully at {tokenizer_cache_path}")
    print(f"Model {model_name} will be downloaded fresh with correct num_labels during training")
    return model_name, tokenizer_cache_path


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
        
        # Return tensors without device assignment - Trainer handles device movement
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }


class TopicClassifierTrainer:
    """Handles training of the topic classification model."""
    
    def __init__(self, num_topics: int, model_name: str = 'distilbert-base-uncased', cache_dir: str = './cached_models'):
        """
        Initialize the trainer.
        
        Args:
            num_topics: Number of topic classes
            model_name: HuggingFace model name
            cache_dir: Directory to cache models
        """
        self.num_topics = num_topics
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Download and cache tokenizer if needed
        self.model_name, self.tokenizer_path = download_and_cache_tokenizer(model_name, cache_dir)
        
        # Load tokenizer from cache
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_path)
        self.model = None
        self.trainer = None
        self.training_history = {'train_loss': [], 'eval_loss': [], 'eval_f1_micro': [], 'eval_cosine_similarity': []}
        
        # Device selection: Prefer MPS (Apple Silicon), then CUDA, then CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
            print("Using Apple Silicon MPS device for training.")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA GPU for training.")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for training.")
    
    def setup_model(self):
        """Initialize the DistilBERT model for multi-label classification."""
        print(f"Loading {self.model_name} with {self.num_topics} topic labels...")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,  # Use original model name, downloads fresh with correct num_labels
            num_labels=self.num_topics,
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)
        
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
        
        # Create logs directory for TensorBoard
        logs_dir = f'{output_dir}/logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=logs_dir,
            logging_steps=10,
            logging_strategy="steps",
            eval_strategy="epoch",  # Fixed: was evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="cosine_similarity",
            greater_is_better=True,
            save_total_limit=2,
            push_to_hub=False,
            report_to=["tensorboard"],  # Enable TensorBoard logging
            dataloader_pin_memory=False,  # Disable for MPS compatibility
        )
        
        # Custom callback to track training history (with fallback if TensorBoard fails)
        try:
            class TrainingHistoryCallback(TensorBoardCallback):
                def __init__(self, trainer_instance):
                    super().__init__()
                    self.trainer_instance = trainer_instance
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    super().on_log(args, state, control, logs, **kwargs)
                    if logs:
                        if 'train_loss' in logs:
                            self.trainer_instance.training_history['train_loss'].append(logs['train_loss'])
                        if 'eval_loss' in logs:
                            self.trainer_instance.training_history['eval_loss'].append(logs['eval_loss'])
                        if 'eval_f1_micro' in logs:
                            self.trainer_instance.training_history['eval_f1_micro'].append(logs['eval_f1_micro'])
                        if 'eval_cosine_similarity' in logs:
                            self.trainer_instance.training_history['eval_cosine_similarity'].append(logs['eval_cosine_similarity'])
        except ImportError:
            # Fallback callback without TensorBoard if there are issues
            from transformers import TrainerCallback
            
            class TrainingHistoryCallback(TrainerCallback):
                def __init__(self, trainer_instance):
                    self.trainer_instance = trainer_instance
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        if 'train_loss' in logs:
                            self.trainer_instance.training_history['train_loss'].append(logs['train_loss'])
                        if 'eval_loss' in logs:
                            self.trainer_instance.training_history['eval_loss'].append(logs['eval_loss'])
                        if 'eval_f1_micro' in logs:
                            self.trainer_instance.training_history['eval_f1_micro'].append(logs['eval_f1_micro'])
                        if 'eval_cosine_similarity' in logs:
                            self.trainer_instance.training_history['eval_cosine_similarity'].append(logs['eval_cosine_similarity'])
            
            print("⚠️  TensorBoard callback not available, using basic logging")
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=10),
                TrainingHistoryCallback(self)
            ]
        )
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        self.trainer.train()
        
        # Save the best model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Create and save training plots
        self._plot_training_history(output_dir)
        
        print(f"Model saved to {output_dir}")
        print(f"Training logs saved to {logs_dir}")
        print(f"View training progress: tensorboard --logdir {logs_dir}")
    
    def _plot_training_history(self, output_dir: str):
        """Create training history plots."""
        os.makedirs(f'{output_dir}/plots', exist_ok=True)
        
        # Get training history from trainer logs
        log_history = self.trainer.state.log_history
        
        # Extract metrics by epoch
        train_losses = []
        eval_losses = []
        eval_f1_scores = []
        eval_cosine_scores = []
        epochs = []
        
        current_epoch = 0
        for log in log_history:
            if 'epoch' in log:
                if 'train_loss' in log:
                    train_losses.append(log['train_loss'])
                    current_epoch = log['epoch']
                
                if 'eval_loss' in log:
                    eval_losses.append(log['eval_loss'])
                    epochs.append(log['epoch'])
                    
                if 'eval_f1_micro' in log:
                    eval_f1_scores.append(log['eval_f1_micro'])
                    
                if 'eval_cosine_similarity' in log:
                    eval_cosine_scores.append(log['eval_cosine_similarity'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training and Validation Loss
        if train_losses and eval_losses:
            train_epochs = np.linspace(1, len(epochs), len(train_losses)) if train_losses else []
            
            ax1.plot(train_epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2, marker='o')
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 Score over epochs
        if eval_f1_scores:
            ax2.plot(epochs, eval_f1_scores, 'g-', label='F1 Micro', linewidth=2, marker='s')
            ax2.set_title('F1 Score (Micro) over Epochs', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('F1 Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cosine Similarity over epochs
        if eval_cosine_scores:
            ax3.plot(epochs, eval_cosine_scores, 'm-', label='Cosine Similarity', linewidth=2, marker='^')
            ax3.set_title('Cosine Similarity over Epochs', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Cosine Similarity')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate Schedule (if available)
        learning_rates = [log.get('learning_rate') for log in log_history if 'learning_rate' in log]
        if learning_rates:
            lr_steps = list(range(len(learning_rates)))
            ax4.plot(lr_steps, learning_rates, 'orange', linewidth=2)
            ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Learning Rate')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nSchedule\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save training metrics to JSON
        training_summary = {
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_eval_loss': eval_losses[-1] if eval_losses else None,
            'best_f1_score': max(eval_f1_scores) if eval_f1_scores else None,
            'best_cosine_similarity': max(eval_cosine_scores) if eval_cosine_scores else None,
            'total_epochs_trained': len(epochs),
            'training_completed': True
        }
        
        with open(f'{output_dir}/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"Training plots saved to {output_dir}/plots/training_history.png")
        print(f"Training summary saved to {output_dir}/training_summary.json")
    
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
        
        # Create detailed results (convert numpy types to native Python types for JSON serialization)
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results = {
            'overall_metrics': convert_numpy_types(metrics),
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
        
        # Also copy training plots to outputs directory for easy access
        training_plots_src = f'{self.trainer.args.output_dir}/plots/training_history.png'
        if os.path.exists(training_plots_src):
            import shutil
            shutil.copy2(training_plots_src, f'{output_dir}/training_history.png')
            print(f"Training plots copied to {output_dir}/training_history.png")
        
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
    parser.add_argument('--cache_dir', type=str, default='./cached_models', help='Directory to cache base models')
    
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
    trainer = TopicClassifierTrainer(num_topics=len(topic_names), cache_dir=args.cache_dir)
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

    # Performance analysis and recommendations
    overall_f1 = results['overall_metrics']['f1_micro']
    cosine_sim = results['overall_metrics']['cosine_similarity']
    
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if overall_f1 < 0.3:
        print("⚠️  LOW PERFORMANCE DETECTED")
        print("\n🔧 RECOMMENDED IMPROVEMENTS:")
        print("1. 📈 INCREASE EPOCHS: Try 10-20 epochs instead of 3")
        print("2. 🎯 LOWER THRESHOLD: Current 0.5 might be too high for multi-label")
        print("3. 📚 MORE DATA: Consider data augmentation or more training samples")
        print("4. ⚙️  TUNE HYPERPARAMETERS: Try different learning rates (1e-5, 5e-5)")
        
        if cosine_sim > 0.15:
            print("✅ Cosine similarity suggests model is learning topic relationships")
        else:
            print("❌ Low cosine similarity - model struggling with topic understanding")
            
        print(f"\n🚀 NEXT TRAINING COMMAND:")
        print(f"python topic_classifier.py --epochs 15 --batch_size {args.batch_size} --learning_rate 1e-5")
            
    elif overall_f1 < 0.6:
        print("📈 MODERATE PERFORMANCE - Room for improvement")
        print("🔧 Consider: More epochs, hyperparameter tuning, or threshold adjustment")
    else:
        print("🎉 GOOD PERFORMANCE! Model is working well.")

    print(f"\n📁 Results saved to: {args.output_dir}")
    print(f"📊 Plots available in: ./outputs/metrics/ and {args.output_dir}/plots/")
    print(f"📈 TensorBoard: tensorboard --logdir {args.output_dir}/logs")

if __name__ == "__main__":
    main() 