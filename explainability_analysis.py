#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SHAP and LIME Explainability Analysis - Adapted for WinoBias
Based on: SHAP_LIME_Analysis.py from HEARTS repository

This script provides model explainability analysis for Part A.5:
- Token-level importance using SHAP
- Token-level importance using LIME  
- Similarity comparison between SHAP and LIME
- Confidence scores for explanations

Modifications:
- Adapted for WinoBias evaluation results
- Uses locally trained model instead of holistic-ai model
- Adjusted column names for WinoBias data structure
"""

import os
import numpy as np
import pandas as pd
import torch
import shap
import re
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================================
# CONFIGURATION - ADAPT TO YOUR SETUP
# ============================================================================

# Input: evaluation results from train.py
RESULTS_FILE = "evaluation_results/albert_winobias_hearts/full_results.csv"

# Model directory: your trained model
MODEL_PATH = "models/albert_winobias_hearts"

# Output directory
OUTPUT_DIR = "explainability_results"

# Sampling parameters (HEARTS uses k=37)
SAMPLE_SIZE_PER_GROUP = 37
RANDOM_SEED = 42

# ============================================================================
# SAMPLING FUNCTION - ADAPTED FOR WINOBIAS
# ============================================================================

def sample_observations(file_path, k, seed):
    """
    Sample observations for explainability analysis.
    
    HEARTS samples by dataset_name and categorisation.
    WinoBias: we sample by group (bias_type) and prediction correctness.
    
    Args:
        file_path (str): Path to full_results.csv
        k (int): Number of samples per group
        seed (int): Random seed
        
    Returns:
        pd.DataFrame: Sampled data
    """
    print(f"\n[INFO] Loading evaluation results from: {file_path}")
    data = pd.read_csv(file_path)
    
    print(f"  Total samples: {len(data)}")
    print(f"  Columns: {data.columns.tolist()}")
    
    # WinoBias structure: text, predicted_label, actual_label, group, dataset_name
    # Sample by group (gender bias type) and prediction correctness
    
    sampled_data = pd.DataFrame()
    
    # Group by dataset type (if available)
    if 'dataset_name' in data.columns:
        groups = data['dataset_name'].unique()
        print(f"  Dataset groups: {groups}")
        
        for group_name in groups:
            group_data = data[data['dataset_name'] == group_name]
            
            # Correct predictions
            correct = group_data[group_data['predicted_label'] == group_data['actual_label']]
            # Incorrect predictions  
            incorrect = group_data[group_data['predicted_label'] != group_data['actual_label']]
            
            # Sample k from each
            if len(correct) >= k:
                correct_sample = correct.sample(n=k, random_state=seed)
            else:
                correct_sample = correct
                
            if len(incorrect) >= k:
                incorrect_sample = incorrect.sample(n=k, random_state=seed)
            else:
                incorrect_sample = incorrect
            
            sampled_data = pd.concat([sampled_data, correct_sample, incorrect_sample], axis=0)
    else:
        # If no dataset_name, sample based on prediction correctness only
        correct = data[data['predicted_label'] == data['actual_label']]
        incorrect = data[data['predicted_label'] != data['actual_label']]
        
        if len(correct) >= k:
            correct_sample = correct.sample(n=k, random_state=seed)
        else:
            correct_sample = correct
            
        if len(incorrect) >= k:
            incorrect_sample = incorrect.sample(n=k, random_state=seed)
        else:
            incorrect_sample = incorrect
        
        sampled_data = pd.concat([sampled_data, correct_sample, incorrect_sample], axis=0)
    
    sampled_data = sampled_data.reset_index(drop=True)
    
    print(f"\n[INFO] Sampled {len(sampled_data)} observations:")
    print(f"  Correct predictions: {len(sampled_data[sampled_data['predicted_label'] == sampled_data['actual_label']])}")
    print(f"  Incorrect predictions: {len(sampled_data[sampled_data['predicted_label'] != sampled_data['actual_label']])}")
    
    return sampled_data

# ============================================================================
# SHAP ANALYSIS - IDENTICAL TO HEARTS
# ============================================================================

def shap_analysis(sampled_data, model_path):
    """
    Compute SHAP values for sampled observations.
    
    This function is identical to HEARTS original code.
    
    Args:
        sampled_data (pd.DataFrame): Sampled observations
        model_path (str): Path to trained model
        
    Returns:
        pd.DataFrame: Token-level SHAP values
    """
    print(f"\n[INFO] Running SHAP analysis...")
    print(f"  Model: {model_path}")
    
    # Load model as pipeline
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    
    # Create SHAP explainer with text masker
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')  
    explainer = shap.Explainer(pipe, masker)
    
    results = []
    class_names = ['LABEL_0', 'LABEL_1']
    
    total = len(sampled_data)
    for index, row in sampled_data.iterrows():
        if (index + 1) % 10 == 0:
            print(f"  Processing {index + 1}/{total}...")
        
        text_input = [row['text']]
        shap_values = explainer(text_input)
        
        # Get SHAP values for stereotype class (LABEL_1)
        label_index = class_names.index("LABEL_1")  
        specific_shap_values = shap_values[:, :, label_index].values
        
        # Extract tokens using same regex as masker
        tokens = re.findall(r'\w+', row['text'])
        
        for token, value in zip(tokens, specific_shap_values[0]):
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_shap': value,
                'sentence': row['text'],
                'group': row.get('group', 'unknown'),
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label'],
                'correct': row['predicted_label'] == row['actual_label']
            })
    
    print(f"  Completed SHAP analysis for {len(results)} tokens")
    return pd.DataFrame(results)

# ============================================================================
# LIME ANALYSIS - IDENTICAL TO HEARTS
# ============================================================================

def custom_tokenizer(text):
    """Tokenizer matching SHAP tokenization"""
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens

def lime_analysis(sampled_data, model_path):
    """
    Compute LIME values for sampled observations.
    
    This function is identical to HEARTS original code.
    
    Args:
        sampled_data (pd.DataFrame): Sampled observations
        model_path (str): Path to trained model
        
    Returns:
        pd.DataFrame: Token-level LIME values
    """
    print(f"\n[INFO] Running LIME analysis...")
    print(f"  Model: {model_path}")
    
    # Load model as pipeline
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    
    # Prediction function for LIME
    def predict_proba(texts):
        preds = pipe(texts, return_all_scores=True)
        probabilities = np.array([[pred['score'] for pred in preds_single] for preds_single in preds])
        return probabilities    
    
    # Create LIME explainer
    explainer = LimeTextExplainer(
        class_names=['LABEL_0', 'LABEL_1'], 
        split_expression=lambda x: custom_tokenizer(x)
    )  
    
    results = []
    total = len(sampled_data)
    
    for index, row in sampled_data.iterrows():
        if (index + 1) % 10 == 0:
            print(f"  Processing {index + 1}/{total}...")
        
        text_input = row['text']
        tokens = custom_tokenizer(text_input)
        
        # Generate LIME explanation
        exp = explainer.explain_instance(
            text_input, 
            predict_proba, 
            num_features=len(tokens), 
            num_samples=100
        )
        
        # Get explanation for stereotype class (label=1)
        explanation_list = exp.as_list(label=1)
        token_value_dict = {token: value for token, value in explanation_list}
        
        for token in tokens:
            value = token_value_dict.get(token, 0)  
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_lime': value,
                'sentence': text_input,
                'group': row.get('group', 'unknown'),
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label'],
                'correct': row['predicted_label'] == row['actual_label']
            })
    
    print(f"  Completed LIME analysis for {len(results)} tokens")
    return pd.DataFrame(results)

# ============================================================================
# SIMILARITY METRICS - IDENTICAL TO HEARTS
# ============================================================================

def compute_cosine_similarity(vector1, vector2):
    """Compute cosine similarity between two vectors"""
    return cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]

def compute_pearson_correlation(vector1, vector2):
    """Compute Pearson correlation between two vectors"""
    correlation, _ = pearsonr(vector1, vector2)
    return correlation

def to_probability_distribution(values):
    """Convert values to probability distribution"""
    min_val = np.min(values)
    if min_val < 0:
        values += abs(min_val)
    total = np.sum(values)
    if total > 0:
        values /= total
    return values

def compute_js_divergence(vector1, vector2):
    """Compute Jensen-Shannon divergence between two distributions"""
    prob1 = to_probability_distribution(vector1.copy())
    prob2 = to_probability_distribution(vector2.copy())
    return jensenshannon(prob1, prob2)

# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def compute_similarity_metrics(shap_df, lime_df, output_dir):
    """
    Compute similarity metrics between SHAP and LIME.
    
    Args:
        shap_df (pd.DataFrame): SHAP results
        lime_df (pd.DataFrame): LIME results
        output_dir (str): Directory to save results
    """
    print(f"\n[INFO] Computing similarity metrics...")
    
    # Sentence-level similarity
    print("  Computing sentence-level similarity...")
    
    # Merge SHAP and LIME by sentence and token
    merged = pd.merge(
        shap_df, 
        lime_df[['sentence_id', 'token', 'value_lime']], 
        on=['sentence_id', 'token'], 
        how='inner'
    )
    
    # Group by sentence
    sentence_similarities = []
    
    for sentence_id in merged['sentence_id'].unique():
        sentence_data = merged[merged['sentence_id'] == sentence_id]
        
        shap_vec = np.array(sentence_data['value_shap'].tolist())
        lime_vec = np.array(sentence_data['value_lime'].tolist())
        
        cos_sim = compute_cosine_similarity(shap_vec, lime_vec)
        pearson = compute_pearson_correlation(shap_vec, lime_vec)
        js_div = compute_js_divergence(shap_vec, lime_vec)
        
        sentence_similarities.append({
            'sentence_id': sentence_id,
            'sentence': sentence_data['sentence'].iloc[0],
            'predicted_label': sentence_data['predicted_label'].iloc[0],
            'actual_label': sentence_data['actual_label'].iloc[0],
            'correct': sentence_data['correct'].iloc[0],
            'cosine_similarity': cos_sim,
            'pearson_correlation': pearson,
            'js_divergence': js_div
        })
    
    similarity_df = pd.DataFrame(sentence_similarities)
    
    # Save results
    similarity_path = os.path.join(output_dir, 'sentence_similarity_metrics.csv')
    similarity_df.to_csv(similarity_path, index=False)
    print(f"  Saved to: {similarity_path}")
    
    # Print summary statistics
    print(f"\n[INFO] Similarity Summary Statistics:")
    print(f"  Cosine Similarity:")
    print(f"    Mean: {similarity_df['cosine_similarity'].mean():.4f}")
    print(f"    Std: {similarity_df['cosine_similarity'].std():.4f}")
    print(f"  Pearson Correlation:")
    print(f"    Mean: {similarity_df['pearson_correlation'].mean():.4f}")
    print(f"    Std: {similarity_df['pearson_correlation'].std():.4f}")
    print(f"  JS Divergence:")
    print(f"    Mean: {similarity_df['js_divergence'].mean():.4f}")
    print(f"    Std: {similarity_df['js_divergence'].std():.4f}")
    
    # Compare correct vs incorrect predictions
    correct_sim = similarity_df[similarity_df['correct'] == True]
    incorrect_sim = similarity_df[similarity_df['correct'] == False]
    
    print(f"\n[INFO] Correct vs Incorrect Predictions:")
    print(f"  Correct predictions (n={len(correct_sim)}):")
    print(f"    Cosine Similarity: {correct_sim['cosine_similarity'].mean():.4f}")
    print(f"  Incorrect predictions (n={len(incorrect_sim)}):")
    print(f"    Cosine Similarity: {incorrect_sim['cosine_similarity'].mean():.4f}")
    
    return similarity_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for explainability analysis.
    """
    
    print("="*70)
    print("SHAP & LIME EXPLAINABILITY ANALYSIS")
    print("Part A.5 - Model Interpretability")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Sample observations
    print("\n" + "="*70)
    print("STEP 1: SAMPLING OBSERVATIONS")
    print("="*70)
    
    sampled_data = sample_observations(RESULTS_FILE, SAMPLE_SIZE_PER_GROUP, RANDOM_SEED)
    
    # Save sampled data
    sample_path = os.path.join(OUTPUT_DIR, 'sampled_data.csv')
    sampled_data.to_csv(sample_path, index=False)
    print(f"  Saved sampled data to: {sample_path}")
    
    # Step 2: SHAP Analysis
    print("\n" + "="*70)
    print("STEP 2: SHAP ANALYSIS")
    print("="*70)
    
    shap_results = shap_analysis(sampled_data, MODEL_PATH)
    
    # Save SHAP results
    shap_path = os.path.join(OUTPUT_DIR, 'shap_results.csv')
    shap_results.to_csv(shap_path, index=False)
    print(f"  Saved SHAP results to: {shap_path}")
    
    # Step 3: LIME Analysis
    print("\n" + "="*70)
    print("STEP 3: LIME ANALYSIS")
    print("="*70)
    
    lime_results = lime_analysis(sampled_data, MODEL_PATH)
    
    # Save LIME results
    lime_path = os.path.join(OUTPUT_DIR, 'lime_results.csv')
    lime_results.to_csv(lime_path, index=False)
    print(f"  Saved LIME results to: {lime_path}")
    
    # Step 4: Compute Similarity Metrics
    print("\n" + "="*70)
    print("STEP 4: SIMILARITY ANALYSIS")
    print("="*70)
    
    similarity_df = compute_similarity_metrics(shap_results, lime_results, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("  - sampled_data.csv")
    print("  - shap_results.csv")
    print("  - lime_results.csv")
    print("  - sentence_similarity_metrics.csv")
    
    print("\nKey Findings:")
    print(f"  Average SHAP-LIME Agreement: {similarity_df['cosine_similarity'].mean():.4f}")
    print(f"  Explanation Confidence: {'High' if similarity_df['cosine_similarity'].mean() > 0.7 else 'Medium'}")
    
    print("\nNext steps:")
    print("  1. Review similarity_metrics.csv for explanation confidence")
    print("  2. Analyze tokens with high SHAP/LIME values")
    print("  3. Include explainability findings in Part B poster")

if __name__ == "__main__":
    main()
    