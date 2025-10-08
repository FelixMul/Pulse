#!/usr/bin/env python3
"""
Pre-download and cache HuggingFace models for Parliament Pulse topic classification.
Run this script once to download models locally and avoid re-downloading during training.
"""

import argparse
from topic_classifier import download_and_cache_tokenizer

def main():
    """Download and cache tokenizer."""
    parser = argparse.ArgumentParser(description='Download and cache HuggingFace tokenizer')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', 
                       help='HuggingFace model name to download tokenizer for')
    parser.add_argument('--cache_dir', type=str, default='./cached_models', 
                       help='Directory to cache tokenizer')
    
    args = parser.parse_args()
    
    print(f"Downloading tokenizer for {args.model_name}...")
    model_name, tokenizer_path = download_and_cache_tokenizer(args.model_name, args.cache_dir)
    print(f"✅ Tokenizer successfully cached at: {tokenizer_path}")
    print(f"🚀 You can now run training with cached tokenizer!")
    print(f"   Command: python topic_classifier.py --cache_dir {args.cache_dir}")
    print(f"📝 Note: Model will download fresh during training with correct architecture")

if __name__ == "__main__":
    main() 