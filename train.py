#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline
from codecarbon import EmissionsTracker

# Enable progress bar and set up logging
os.environ["HUGGINGFACE_TRAINER_ENABLE_PROGRESS_BAR"] = "1"
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

# ============================================================================
# DATA LOADING - ADAPTED FOR WINOBIAS
# ============================================================================

def load_winobias_data(train_path, valid_path, test_path):
    """
    Load preprocessed WinoBias data and adapt to HEARTS format.
    
    HEARTS expects columns: ['text', 'label', 'group']
    WinoBias has columns: ['sentence', 'label', 'bias_type', 'dataset_type', 'stereotype_type']
    
    Args:
        train_path (str): Path to train.csv
        valid_path (str): Path to valid.csv  
        test_path (str): Path to test.csv
        
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames
    """
    print("[INFO] Loading WinoBias data...")
    
    # Load CSVs
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    # Rename columns to match HEARTS format
    train_df = train_df.rename(columns={
        'sentence': 'text',
        'bias_type': 'group'
    })
    valid_df = valid_df.rename(columns={
        'sentence': 'text',
        'bias_type': 'group'
    })
    test_df = test_df.rename(columns={
        'sentence': 'text',
        'bias_type': 'group'
    })
    
    # Add dataset name for consistency with HEARTS
    train_df['data_name'] = 'winobias'
    valid_df['data_name'] = 'winobias'
    test_df['data_name'] = 'winobias'
    
    # Select only required columns
    required_cols = ['text', 'label', 'group', 'data_name']
    train_data = train_df[required_cols]
    val_data = valid_df[required_cols]
    test_data = test_df[required_cols]
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Valid: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"\nFirst few examples from training data:")
    print(train_data.head(3))
    
    return train_data, val_data, test_data

# ============================================================================
# TRAINING FUNCTION - IDENTICAL TO HEARTS
# ============================================================================

def train_model(train_data, val_data, model_path, batch_size, epoch, learning_rate, 
                model_output_dir, seed=42):
    """
    Train model using HEARTS methodology.
    
    This function is identical to HEARTS original code.
    
    Args:
        train_data (pd.DataFrame): Training data with columns ['text', 'label', 'group']
        val_data (pd.DataFrame): Validation data
        model_path (str): HuggingFace model path (e.g., 'albert/albert-base-v2')
        batch_size (int): Batch size (HEARTS: 64)
        epoch (int): Number of epochs (HEARTS: 6)
        learning_rate (float): Learning rate (HEARTS: 2e-5)
        model_output_dir (str): Directory to save trained model
        seed (int): Random seed
        
    Returns:
        str: Path to saved model
    """
    
    np.random.seed(seed)
    num_labels = len(train_data['label'].unique())
    print(f"\n[INFO] Training Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Number of labels: {num_labels}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epoch}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output directory: {model_output_dir}")
    
    # Start carbon tracking (HEARTS emphasis on sustainability)
    tracker = EmissionsTracker()
    tracker.start()
    
    # Load model and tokenizer
    print(f"\n[INFO] Loading model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Handle GPT models (if used)
    if model_path.startswith("gpt"):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    # Convert to HuggingFace Dataset format
    print("[INFO] Tokenizing datasets...")
    tokenized_train = Dataset.from_pandas(train_data).map(
        tokenize_function, batched=True
    ).map(lambda examples: {'labels': examples['label']})
    
    tokenized_val = Dataset.from_pandas(val_data).map(
        tokenize_function, batched=True
    ).map(lambda examples: {'labels': examples['label']})
    
    print(f"  Sample tokenized input from train: {tokenized_train[0]}")
    
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
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Training arguments (HEARTS configuration)
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epoch,
        eval_strategy="epoch",  
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_steps=100,
        seed=seed
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        #tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("\n[INFO] Starting training...")
    trainer.train()
    
    # Save model
    print(f"[INFO] Saving model to {model_output_dir}")
    trainer.save_model(model_output_dir)
    
    tokenizer.save_pretrained(model_output_dir)
    # Stop carbon tracking
    emissions = tracker.stop()
    print(f"\n[SUCCESS] Training complete!")
    if emissions is None:
        print("  Estimated CO2 emissions: N/A (codecarbon tracker did not return a value)")
    else:
        print(f"  Estimated CO2 emissions: {emissions:.6f} kg")
    
    return model_output_dir

# ============================================================================
# EVALUATION FUNCTION - IDENTICAL TO HEARTS
# ============================================================================

def evaluate_model(test_data, model_output_dir, result_output_dir, seed=42):
    """
    Evaluate trained model on test data.
    
    This function is identical to HEARTS original code.
    
    Args:
        test_data (pd.DataFrame): Test data with columns ['text', 'label', 'group']
        model_output_dir (str): Directory where trained model is saved
        result_output_dir (str): Directory to save evaluation results
        seed (int): Random seed
        
    Returns:
        pd.DataFrame: Classification report as DataFrame
    """
    
    np.random.seed(seed)
    num_labels = len(test_data['label'].unique())
    print(f"\n[INFO] Evaluation Configuration:")
    print(f"  Model directory: {model_output_dir}")
    print(f"  Number of labels: {num_labels}")
    print(f"  Test samples: {len(test_data)}")
    
    # Load trained model
    print(f"[INFO] Loading trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    
    # Handle GPT models (if used)
    if model_output_dir.startswith("gpt"):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)
    
    # Tokenize test data
    print("[INFO] Tokenizing test data...")
    tokenized_test = Dataset.from_pandas(test_data).map(
        tokenize_function, batched=True
    ).map(lambda examples: {'labels': examples['label']})
    
    print(f"  Sample tokenized input: {tokenized_test[0]}")
    
    # Create output directory
    os.makedirs(result_output_dir, exist_ok=True)
    
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
    
    # Save full results
    results_file_path = os.path.join(result_output_dir, "full_results.csv")
    results_df.to_csv(results_file_path, index=False)
    print(f"[INFO] Saved full results to: {results_file_path}")
    
    # Generate classification report
    print("\n[INFO] Generating classification report...")
    report = classification_report(y_true, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_file_path = os.path.join(result_output_dir, "classification_report.csv")
    df_report.to_csv(report_file_path)
    print(f"[INFO] Saved classification report to: {report_file_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall: {report['macro avg']['recall']:.4f}")
    print(f"  F1-score: {report['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print("\nPer-class metrics:")
    print(df_report)
    
    return df_report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for Part A.4 and A.5.
    """
    
    print("="*70)
    print("HEARTS REPLICATION - WINOBIAS ADAPTATION")
    print("="*70)
    
    # Configuration
    DATA_DIR = "dataset_winobias"
    MODEL_OUTPUT_DIR = "models/albert_winobias_hearts"
    RESULT_OUTPUT_DIR = "evaluation_results/albert_winobias_hearts"
    
    # HEARTS hyperparameters 
    MODEL_PATH = "albert/albert-base-v2"  
    BATCH_SIZE = 32                        
    EPOCHS = 3                             
    LEARNING_RATE = 1e-5                   
    SEED = 42                             
    
    # Part A.4: Load data and train model
    print("\n" + "="*70)
    print("PART A.4: MODEL TRAINING")
    print("="*70)
    
    train_data, val_data, test_data = load_winobias_data(
        train_path=f"{DATA_DIR}/train.csv",
        valid_path=f"{DATA_DIR}/valid.csv",
        test_path=f"{DATA_DIR}/test.csv"
    )
    
    model_dir = train_model(
        train_data=train_data,
        val_data=val_data,
        model_path=MODEL_PATH,
        batch_size=BATCH_SIZE,
        epoch=EPOCHS,
        learning_rate=LEARNING_RATE,
        model_output_dir=MODEL_OUTPUT_DIR,
        seed=SEED
    )
    
    # Part A.5: Evaluate model
    print("\n" + "="*70)
    print("PART A.5: MODEL EVALUATION")
    print("="*70)
    
    report = evaluate_model(
        test_data=test_data,
        model_output_dir=model_dir,
        result_output_dir=RESULT_OUTPUT_DIR,
        seed=SEED
    )
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"\nModel saved to: {MODEL_OUTPUT_DIR}")
    print(f"Results saved to: {RESULT_OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - {RESULT_OUTPUT_DIR}/full_results.csv")
    print(f"  - {RESULT_OUTPUT_DIR}/classification_report.csv")
    print("\nNext steps:")
    print("  1. Review classification report for metrics")
    print("  2. Compare with HEARTS baseline (F1: 0.815)")
    print("  3. Prepare Part B poster with results")

if __name__ == "__main__":
    main()