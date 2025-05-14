#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_A_RESULTS = os.path.join(SCRIPT_DIR, 'model_A', 'results')
MODEL_B_RESULTS = os.path.join(SCRIPT_DIR, 'model_B', 'results')

def find_latest_results(model_results_path):
    """Find latest results directory for a model"""
    result_dirs = glob(os.path.join(model_results_path, '*'))
    if not result_dirs:
        print(f"No results found in {model_results_path}")
        return None
    
    # Sort by creation time (newest last)
    result_dirs.sort(key=os.path.getctime)
    return result_dirs[-1]

def load_metrics(results_dir):
    """Load metrics from YOLOv5 results directory"""
    metrics = {}
    
    # Load training metrics from results.csv
    results_csv = os.path.join(results_dir, 'results.csv')
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        metrics['precision'] = df['precision'].iloc[-1]
        metrics['recall'] = df['recall'].iloc[-1]
        metrics['mAP_0.5'] = df['mAP_0.5'].iloc[-1]
        metrics['mAP_0.5:0.95'] = df['mAP_0.5:0.95'].iloc[-1]
        
        # Calculate training time (in hours)
        metrics['training_time'] = len(df) * df['time'].mean() / 3600
    
    # Check for specialized small object metrics if available
    small_object_metrics = os.path.join(results_dir, 'small_object_metrics.json')
    if os.path.exists(small_object_metrics):
        with open(small_object_metrics, 'r') as f:
            metrics['small_objects'] = json.load(f)
    
    return metrics

def plot_comparison(model_a_metrics, model_b_metrics):
    """Generate comparison plots between models"""
    # Ensure output directory exists
    output_dir = os.path.join(SCRIPT_DIR, 'comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Key metrics comparison
    metrics = ['precision', 'recall', 'mAP_0.5', 'mAP_0.5:0.95']
    model_a_values = [model_a_metrics[m] for m in metrics]
    model_b_values = [model_b_metrics[m] for m in metrics]
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, model_a_values, width, label='Model A (Original)')
    ax.bar(x + width/2, model_b_values, width, label='Model B (Preprocessed)')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    
    # Training time comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Model A', 'Model B'], 
           [model_a_metrics['training_time'], model_b_metrics['training_time']])
    ax.set_ylabel('Training Time (hours)')
    ax.set_title('Training Duration Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'))
    
    return output_dir

def generate_report(model_a_metrics, model_b_metrics, output_dir):
    """Generate a text report comparing the models"""
    report_path = os.path.join(output_dir, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# YOLOv5m Model Comparison Report\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Model A (Original) | Model B (Preprocessed) | Difference | % Improvement |\n")
        f.write("|--------|-------------------|------------------------|------------|---------------|\n")
        
        for metric in ['precision', 'recall', 'mAP_0.5', 'mAP_0.5:0.95']:
            val_a = model_a_metrics[metric]
            val_b = model_b_metrics[metric]
            diff = val_b - val_a
            pct_change = (diff / val_a) * 100 if val_a != 0 else float('inf')
            
            f.write(f"| {metric} | {val_a:.4f} | {val_b:.4f} | {diff:.4f} | {pct_change:.2f}% |\n")
        
        # Training time
        time_a = model_a_metrics['training_time']
        time_b = model_b_metrics['training_time']
        time_diff = time_b - time_a
        time_pct = (time_diff / time_a) * 100 if time_a != 0 else float('inf')
        
        f.write(f"| Training Time (hours) | {time_a:.2f} | {time_b:.2f} | {time_diff:.2f} | {time_pct:.2f}% |\n\n")
        
        # Analysis and conclusion
        f.write("## Analysis\n\n")
        
        # Determine which model performed better overall
        map_comparison = "Model B (Preprocessed)" if model_b_metrics['mAP_0.5'] > model_a_metrics['mAP_0.5'] else "Model A (Original)"
        
        f.write(f"- **Overall Detection Performance**: {map_comparison} demonstrated better overall detection performance as measured by mAP@0.5.\n")
        
        # Small object detection comparison if available
        if 'small_objects' in model_a_metrics and 'small_objects' in model_b_metrics:
            small_a = model_a_metrics['small_objects']['mAP_small']
            small_b = model_b_metrics['small_objects']['mAP_small']
            small_better = "Model B (Preprocessed)" if small_b > small_a else "Model A (Original)"
            
            f.write(f"- **Small Object Detection**: {small_better} performed better at detecting small and distant objects.\n")
        else:
            f.write("- **Small Object Detection**: Specialized metrics for small objects were not available.\n")
        
        # Training efficiency
        time_better = "Model B (Preprocessed)" if time_b < time_a else "Model A (Original)"
        f.write(f"- **Training Efficiency**: {time_better} was more efficient in terms of training time.\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Based on the performance metrics above, ")
        
        # General conclusion
        if model_b_metrics['mAP_0.5'] > model_a_metrics['mAP_0.5']:
            f.write("the preprocessed dataset (Model B) shows improvements in detection accuracy ")
        else:
            f.write("the original dataset (Model A) performs better in detection accuracy ")
            
        if time_b < time_a:
            f.write("and requires less training time, ")
        else:
            f.write("though it requires more training time, ")
            
        f.write("suggesting that ")
        
        if model_b_metrics['mAP_0.5'] > model_a_metrics['mAP_0.5']:
            f.write("the preprocessing techniques applied to the dataset have positively impacted model performance.")
        else:
            f.write("the preprocessing techniques applied may not have benefited this particular model configuration.")
    
    return report_path

def main():
    print("=== YOLOv5 Model Comparison Tool ===")
    
    # Find most recent results
    model_a_dir = find_latest_results(MODEL_A_RESULTS)
    model_b_dir = find_latest_results(MODEL_B_RESULTS)
    
    if not model_a_dir or not model_b_dir:
        print("Couldn't find results for both models. Please ensure both models have been trained.")
        return
    
    print(f"Using results from:\n- Model A: {model_a_dir}\n- Model B: {model_b_dir}")
    
    # Load metrics
    model_a_metrics = load_metrics(model_a_dir)
    model_b_metrics = load_metrics(model_b_dir)
    
    if not model_a_metrics or not model_b_metrics:
        print("Failed to load metrics for one or both models.")
        return
    
    # Generate visualizations
    output_dir = plot_comparison(model_a_metrics, model_b_metrics)
    
    # Generate report
    report_path = generate_report(model_a_metrics, model_b_metrics, output_dir)
    
    print(f"Comparison completed! Report saved to {report_path}")
    print(f"Visualizations saved to {output_dir}")
    
if __name__ == "__main__":
    main() 