import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_per_class_f1(csv_path: str, output_path: str):
    """Generates a bar chart of per-class F1 scores comparing models."""
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df, x='Class', y='Per-Class F1', hue='Model')
    plt.xticks(rotation=90)
    plt.title('Per-Class F1 Score Comparison')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved bar chart to {output_path}")

def plot_summary_metrics(results_txt: str, output_path: str):
    """Parses results.txt and plots overall Accuracy and F1."""
    models = []
    accs = []
    f1s = []
    
    import re
    with open(results_txt, 'r') as f:
        lines = f.readlines()
        current_model = ""
        captured_for_current = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.endswith(':'):
                current_model = line.strip(':').strip()
                captured_for_current = False
                continue
            
            if captured_for_current:
                continue

            # Look for Accuracy and F1 in the same line
            acc_match = re.search(r'(?:ACC|acc|val_acc)=([\d.]+)', line)
            f1_match = re.search(r'(?:F1|f1|val_f1)=([\d.]+)', line)
            
            if acc_match and f1_match:
                accs.append(float(acc_match.group(1)))
                f1s.append(float(f1_match.group(1)))
                models.append(current_model)
                captured_for_current = True

    data = []
    for i in range(len(models)):
        data.append({'Model': models[i], 'Metric': 'Accuracy', 'Value': accs[i]})
        data.append({'Model': models[i], 'Metric': 'F1 Score', 'Value': f1s[i]})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Value', hue='Metric')
    plt.title('Overall Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)  # Added some headroom for labels
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved summary chart to {output_path}")

if __name__ == "__main__":
    import os
    if os.path.exists('results/per_class_metrics_pooled.csv'):
        plot_per_class_f1('results/per_class_metrics_pooled.csv', 'results/per_class_f1.png')
    if os.path.exists('results/results.txt'):
        plot_summary_metrics('results/results.txt', 'results/overall_comparison.png')
