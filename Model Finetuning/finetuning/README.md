# Model Finetuning for Parliament Pulse

## 🎯 Purpose

This directory contains the machine learning pipeline for training custom NLP models on our synthetic constituent email dataset. Instead of using generic, unsupervised tools like BERTopic or vaderSentiment, we leverage our labeled dataset of 1,250 synthetic emails to train highly accurate, domain-specific models.

## 📊 Dataset Overview

Our dataset (`email_dataset_final.csv`) contains:
- **1,250 synthetic emails** simulating a UK MP's constituent inbox
- **Multiple labels per email**: topics, sentiment, persona, length
- **Realistic content**: Generated via LLM and post-processed for authenticity
- **Multi-label topics**: Emails can have 1-3 topics from predefined categories

## 🎯 Training Objectives

### 1. Topic Classification Model
- **Architecture**: Fine-tuned DistilBERT for multi-label classification
- **Input**: Raw email text
- **Output**: Probability scores for each topic category
- **Evaluation**: Cosine similarity between predicted and true topic vectors

### 2. Sentiment Analysis Model
- **Architecture**: Fine-tuned DistilBERT for multi-class classification
- **Input**: Raw email text
- **Output**: Sentiment classification (Very Negative, Negative, Neutral, Positive, Very Positive, Mixed)
- **Evaluation**: Accuracy, F1-score, confusion matrix

## 🏗️ Model Architecture

Both models use **DistilBERT-base-uncased** as the foundation:

```
Input Text → DistilBERT Encoder → [CLS] Token → Classification Head → Predictions
```

### Topic Model (Multi-label)
- Final layer: Sigmoid activation for each topic
- Loss function: Binary Cross-Entropy with Logits
- Threshold: 0.5 for binary topic assignment

### Sentiment Model (Multi-class)
- Final layer: Softmax activation
- Loss function: Cross-Entropy
- Output: Single sentiment class with confidence scores

## 📁 File Structure

```
finetuning/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data_preprocessing.py        # Dataset loading and preprocessing
├── topic_classifier.py         # Topic classification training
├── sentiment_classifier.py     # Sentiment analysis training
├── evaluation.py               # Model evaluation utilities
├── models/                     # Saved trained models
│   ├── topic_model/            # Fine-tuned topic classifier
│   └── sentiment_model/        # Fine-tuned sentiment classifier
├── notebooks/                  # Jupyter notebooks for EDA and experiments
│   ├── data_exploration.ipynb  # Dataset exploration and visualization
│   └── model_experiments.ipynb # Model training experiments
└── outputs/                    # Training logs, plots, metrics
    ├── logs/                   # Training logs
    ├── plots/                  # Visualization outputs
    └── metrics/                # Evaluation results
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd Model\ Finetuning/finetuning
pip install -r requirements.txt
```

### 2. Data Exploration
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 3. Train Topic Classifier
```bash
python topic_classifier.py --epochs 3 --batch_size 16 --learning_rate 2e-5
```

### 4. Train Sentiment Classifier
```bash
python sentiment_classifier.py --epochs 3 --batch_size 16 --learning_rate 2e-5
```

### 5. Evaluate Models
```bash
python evaluation.py --model_type topic
python evaluation.py --model_type sentiment
```

## 🎯 Expected Performance

Based on similar fine-tuning tasks on domain-specific data:

### Topic Classification
- **Target F1-Score**: >0.85 (macro-averaged)
- **Target Cosine Similarity**: >0.90
- **Multi-label accuracy**: Ability to correctly identify 1-3 topics per email

### Sentiment Analysis
- **Target Accuracy**: >0.90
- **Target F1-Score**: >0.85 (macro-averaged)
- **Robustness**: Handle varied writing styles, formality levels, and edge cases

## 🔄 Integration with Main Application

Once trained, models will be:
1. **Saved** in HuggingFace format for easy loading
2. **Integrated** into `backend/app/nlp_processor.py`
3. **Deployed** via FastAPI endpoints for real-time inference
4. **Cached** for optimal performance in the dashboard

## 📈 Future Enhancements

1. **Active Learning**: Incrementally improve models with new labeled data
2. **Model Distillation**: Create even smaller, faster models for deployment
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Cross-validation**: Robust evaluation across different data splits

## 🛠️ Technical Notes

- **GPU Recommended**: Training will be faster with CUDA-enabled GPU
- **Memory Requirements**: ~4-8GB RAM for training, ~1GB for inference
- **Training Time**: ~30-60 minutes per model on modern hardware
- **Model Size**: ~250MB per fine-tuned DistilBERT model 