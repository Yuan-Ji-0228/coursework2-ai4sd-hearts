#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part A.1: HEARTS Baseline Replication - ALBERT-V2 on EMGSD
Reproduces HEARTS paper results: Macro F1 = 0.815

Based on: BERT_Models_Fine_Tuning.py from HEARTS repository
Target: Replicate original HEARTS performance on EMGSD dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline
from codecarbon import EmissionsTracker

# ============================================================================
# CONFIGURATION - HEARTS ORIGINAL SETUP
# ============================================================================

# HEARTS hyperparameters (DO NOT CHANGE)
MODEL_PATH = "albert/albert-base-v2"
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
EPOCHS = 6
SEED = 42
MAX_LENGTH = 512

# Output directories
MODEL_OUTPUT_DIR = "models/hearts_emgsd_baseline"
EVAL_OUTPUT_DIR = "evaluation_results/hearts_emgsd_baseline"

# ============================================================================
# DATA LOADING - EMGSD FROM HUGGINGFACE
# ============================================================================

def load_emgsd_data():
    """
    Load EMGSD dataset from HuggingFace.
    
    EMGSD Structure:
    - text: The text sample
    - category: "stereotype", "neutral", "unrelated"
    - label: Binary (0=non-stereotype, 1=stereotype)
    - dimensions: age, disability, gender, nationality, race, religion, sexuality
    
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames
    """
    
    print("\n[INFO] Loading EMGSD dataset from HuggingFace...")
    print("  This may take a few minutes (57,201 samples)...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("holistic-ai/EMGSD")
    
    # EMGSD only has 'train' split, so we need to create splits
    full_data = dataset['train'].to_pandas()
    
    print(f"\n[INFO] EMGSD Dataset Statistics:")
    print(f"  Total samples: {len(full_data)}")
    print(f"  Columns: {full_data.columns.tolist()}")
    
    # Convert category to binary label
    # stereotype -> 1, neutral/unrelated -> 0
    full_data['label'] = full_data['category'].apply(
        lambda x: 1 if x.strip().lower() == 'stereotype' else 0
    )
    
    # Add group column (use 'dimensions' column if available, otherwise use category)
    if 'dimensions' in full_data.columns:
        full_data['group'] = full_data['dimensions']
    else:
        full_data['group'] = 'general'
    
    # Add dataset name for consistency
    full_data['data_name'] = 'emgsd'
    
    # Select required columns
    required_cols = ['text', 'label', 'group', 'data_name']
    data = full_data[required_cols].copy()
    
    print(f"\n[INFO] Label Distribution:")
    print(data['label'].value_counts())
    
    # Split: 80% train, 10% validation, 10% test (stratified)
    train_data, temp_data = train_test_split(
        data, 
        test_size=0.2, 
        random_state=SEED, 
        stratify=data['label']
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=SEED, 
        stratify=temp_data['label']
    )
    
    print(f"\n[INFO] Data Splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    print(f"\n[INFO] First few training examples:")
    print(train_data.head(3))
    
    return train_data, val_data, test_data

# ============================================================================
# TRAINING FUNCTION - IDENTICAL TO HEARTS
# ============================================================================

def train_hearts_baseline(train_data, val_data):
    """
    Train ALBERT-V2 on EMGSD using HEARTS methodology.
    
    This replicates the original HEARTS training setup exactly.
    
    Args:
        train_data (pd.DataFrame): Training data
        val_data (pd.DataFrame): Validation data
        
    Returns:
        str: Path to saved model
    """
    
    print("\n" + "="*70)
    print("TRAINING HEARTS BASELINE - ALBERT-V2 ON EMGSD")
    print("="*70)
    
    np.random.seed(SEED)
    num_labels = 2  # Binary classification
    
    print(f"\n[INFO] Training Configuration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Number of labels: {num_labels}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output directory: {MODEL_OUTPUT_DIR}")
    
    # Start carbon tracking
    tracker = EmissionsTracker()
    tracker.start()
    
    # Load model and tokenizer
    print(f"\n[INFO] Loading ALBERT-V2 model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=MAX_LENGTH)
    
    # Convert to HuggingFace Dataset format
    print("[INFO] Tokenizing datasets...")
    tokenized_train = Dataset.from_pandas(train_data).map(
        tokenize_function, batched=True
    ).map(lambda examples: {'labels': examples['label']})
    
    tokenized_val = Dataset.from_pandas(val_data).map(
        tokenize_function, batched=True
    ).map(lambda examples: {'labels': examples['label']})
    
    print(f"  Sample tokenized input: {tokenized_train[0]}")
    
    # Compute metrics function (HEARTS uses macro F1)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "balanced accuracy": balanced_acc
        }
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


    # Training arguments (HEARTS configuration)
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_steps=100,
        seed=SEED,
        no_cuda=False,  
        report_to="none",     
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("\n[INFO] Starting training...")
    print(f"  Expected duration: ~2-3 hours (GPU), ~10-12 hours (CPU)")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    trainer.train()
    
    # Save model
    print(f"\n[INFO] Saving model to {MODEL_OUTPUT_DIR}")
    trainer.save_model(MODEL_OUTPUT_DIR)
    
    # Stop carbon tracking
    emissions = tracker.stop()
    print(f"\n[SUCCESS] Training complete!")
    print(f"  Estimated CO2 emissions: {emissions:.6f} kg")
    
    return MODEL_OUTPUT_DIR

# ============================================================================
# EVALUATION FUNCTION - IDENTICAL TO HEARTS
# ============================================================================

def evaluate_hearts_baseline(test_data, model_dir):
    """
    Evaluate trained model on EMGSD test set.
    
    Args:
        test_data (pd.DataFrame): Test data
        model_dir (str): Directory where trained model is saved
        
    Returns:
        pd.DataFrame: Classification report
    """
    
    print("\n" + "="*70)
    print("EVALUATING HEARTS BASELINE")
    print("="*70)
    
    np.random.seed(SEED)
    
    # Load trained model
    print(f"\n[INFO] Loading trained model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Create prediction pipeline
    print("[INFO] Generating predictions...")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
    
    # Get predictions
    predictions = pipe(test_data['text'].to_list(), return_all_scores=True)
    pred_labels = [
        int(max(pred, key=lambda x: x['score'])['label'].split('_')[-1]) 
        for pred in predictions
    ]
    pred_probs = [
        max(pred, key=lambda x: x['score'])['score'] 
        for pred in predictions
    ]
    y_true = test_data['label'].tolist()
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'],
        'dataset_name': test_data['data_name']
    })
    
    # Create output directory
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    
    # Save full results
    results_path = os.path.join(EVAL_OUTPUT_DIR, "full_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] Saved full results to: {results_path}")
    
    # Generate classification report
    print("\n[INFO] Generating classification report...")
    report = classification_report(y_true, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_path = os.path.join(EVAL_OUTPUT_DIR, "classification_report.csv")
    df_report.to_csv(report_path)
    print(f"[INFO] Saved classification report to: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS - HEARTS BASELINE REPLICATION")
    print("="*70)
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall: {report['macro avg']['recall']:.4f}")
    print(f"  F1-score: {report['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    
    print(f"\n** HEARTS Paper Target: F1 = 0.815 **")
    print(f"** Our Replication: F1 = {report['macro avg']['f1-score']:.3f} **")
    
    diff = abs(report['macro avg']['f1-score'] - 0.815)
    if diff <= 0.05:
        print(f"** ✅ Within ±5% of original paper (diff: {diff:.3f}) **")
    else:
        print(f"** ⚠️ Outside ±5% tolerance (diff: {diff:.3f}) **")
    
    print("\nPer-class metrics:")
    print(df_report)
    
    return df_report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for Part A.1 baseline replication.
    """
    
    print("="*70)
    print("PART A.1: HEARTS BASELINE REPLICATION")
    print("ALBERT-V2 on EMGSD Dataset")
    print("Target: Reproduce F1 = 0.815 (±5%)")
    print("="*70)
    
    # Load EMGSD data
    train_data, val_data, test_data = load_emgsd_data()
    
    # Train model
    model_dir = train_hearts_baseline(train_data, val_data)
    
    # Evaluate model
    report = evaluate_hearts_baseline(test_data, model_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("BASELINE REPLICATION COMPLETED!")
    print("="*70)
    print(f"\nModel saved to: {MODEL_OUTPUT_DIR}")
    print(f"Results saved to: {EVAL_OUTPUT_DIR}")
    
    macro_f1 = report.loc['macro avg', 'f1-score']
    print(f"\n✅ Part A.1 Complete: Baseline Macro F1 = {macro_f1:.3f}")
    print(f"   HEARTS Paper: Macro F1 = 0.815")
    print(f"   Difference: {abs(macro_f1 - 0.815):.3f}")
    
    print("\nNext steps:")
    print("  1. ✅ Part A.1 complete (baseline replication)")
    print("  2. → Part A.2: Write problem statement")
    print("  3. → Part A.3: Preprocess WinoBias dataset")
    print("  4. → Part A.4: Train ALBERT-V2 on WinoBias")
    print("  5. → Part A.5: Compare EMGSD vs WinoBias performance")

if __name__ == "__main__":
    main()