import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaTokenizer, 
    LlamaForSequenceClassification,
    LlamaConfig,
    TrainingArguments, 
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import argparse
import json
import os
from dataclasses import dataclass


class ClassificationDataset(Dataset):
    """Custom dataset for text classification tasks."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def load_data(data_path: str, task_type: str = 'binary'):
    """
    Load data from CSV file.
    Expected format: CSV with 'text' and 'label' columns
    
    Args:
        data_path: Path to CSV file
        task_type: 'binary' or 'multiclass'
    
    Returns:
        texts, labels, num_labels
    """
    df = pd.read_csv(data_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Convert labels to integers if they're strings
    if isinstance(labels[0], str):
        unique_labels = list(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_to_id[label] for label in labels]
        
        # Save label mapping
        with open(f'label_mapping_{task_type}.json', 'w') as f:
            json.dump(label_to_id, f)
    
    num_labels = len(set(labels))
    
    # Validate task type
    if task_type == 'binary' and num_labels != 2:
        raise ValueError(f"Binary classification requires 2 labels, found {num_labels}")
    elif task_type == 'multiclass' and num_labels < 3:
        raise ValueError(f"Multiclass classification requires 3+ labels, found {num_labels}")
    
    return texts, labels, num_labels


def create_model_and_tokenizer(model_name: str, num_labels: int, task_type: str):
    """Create model and tokenizer for classification."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False  # Disable caching during training
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    # Set pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def train_classification_model(
    model_name: str,
    train_data_path: str,
    val_data_path: Optional[str],
    task_type: str,
    output_dir: str,
    max_length: int = 512,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100
):
    """
    Train a Llama-3 model for classification.
    
    Args:
        model_name: Hugging Face model name (e.g., 'meta-llama/Meta-Llama-3-8B')
        train_data_path: Path to training data CSV
        val_data_path: Path to validation data CSV (optional)
        task_type: 'binary' or 'multiclass'
        output_dir: Directory to save the fine-tuned model
        max_length: Maximum sequence length
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay
        gradient_accumulation_steps: Gradient accumulation steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
    """
    
    print(f"Starting {task_type} classification training...")
    
    # Load data
    train_texts, train_labels, num_labels = load_data(train_data_path, task_type)
    print(f"Loaded {len(train_texts)} training samples with {num_labels} labels")
    
    val_texts, val_labels = None, None
    if val_data_path:
        val_texts, val_labels, _ = load_data(val_data_path, task_type)
        print(f"Loaded {len(val_texts)} validation samples")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_name, num_labels, task_type)
    
    # Create datasets
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = None
    if val_texts and val_labels:
        val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_f1" if val_dataset else None,
        greater_is_better=True,
        learning_rate=learning_rate,
        fp16=True,  # Use mixed precision for faster training
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
        save_total_limit=2,  # Keep only 2 checkpoints
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics if val_dataset else None,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        'task_type': task_type,
        'num_labels': num_labels,
        'model_name': model_name,
        'max_length': max_length,
        'training_samples': len(train_texts),
        'validation_samples': len(val_texts) if val_texts else 0
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Training completed! Model saved to {output_dir}")
    
    # Final evaluation if validation set exists
    if val_dataset:
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate()
        print("Final evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3 8B for classification")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Hugging Face model name")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data CSV")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data CSV")
    parser.add_argument("--task_type", type=str, choices=['binary', 'multiclass'], required=True,
                        help="Classification task type")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train_classification_model(
        model_name=args.model_name,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        task_type=args.task_type,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


if __name__ == "__main__":
    main()

