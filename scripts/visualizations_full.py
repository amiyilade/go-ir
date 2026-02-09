#!/usr/bin/env python3
"""
Script 9 (FULL): Generate Visualizations for Paper
Creates publication-quality figures for all research questions.

FULL DATASET VERSION - Includes NEW figures for cross-model comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Configuration
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "visualizations"

def load_alignment_results():
    """Load RQ1 alignment results."""
    file_path = RESULTS_DIR / "ast_alignment" / "all_alignment_results.json"
    with open(file_path, 'r') as f:
        return json.load(f)

def load_probing_results():
    """Load RQ2 probing results."""
    file_path = RESULTS_DIR / "structural_probing" / "all_probing_results.json"
    with open(file_path, 'r') as f:
        return json.load(f)

def load_tree_induction_results():
    """Load RQ3 tree induction results."""
    file_path = RESULTS_DIR / "tree_induction" / "all_tree_induction_results.json"
    with open(file_path, 'r') as f:
        return json.load(f)

def load_construct_results():
    """Load construct analysis results."""
    try:
        file_path = RESULTS_DIR / "construct_analysis" / "all_construct_analysis.json"
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_cross_model_results():
    """Load cross-model analysis results."""
    try:
        file_path = RESULTS_DIR / "cross_model_analysis" / "all_cross_model_analysis.json"
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def plot_alignment_heatmap(data, output_dir):
    """Figure 1: Layer-Head Alignment Heatmap (RQ1) - Multi-Model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models_tasks = [
        ('unixcoder_code_to_text', 'UniXcoder\nCode-to-Text', axes[0, 0]),
        ('unixcoder_code_to_code', 'UniXcoder\nCode-to-Code', axes[0, 1]),
        ('codebert_code_to_text', 'CodeBERT\nCode-to-Text', axes[1, 0]),
        ('codebert_code_to_code', 'CodeBERT\nCode-to-Code', axes[1, 1])
    ]
    
    for task_key, title, ax in models_tasks:
        if task_key not in data:
            continue
            
        results = data[task_key]['all_results']
        
        layers = 12
        heads = 12
        alignment_matrix = np.zeros((layers, heads))
        
        for result in results:
            layer = result['layer']
            head = result['head']
            score = result['mean_alignment_score']
            alignment_matrix[layer, head] = score
        
        im = ax.imshow(alignment_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.75)
        
        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')
        ax.set_title(title)
        ax.set_xticks(range(heads))
        ax.set_yticks(range(layers))
        
        # Highlight best head (Layer 7, Head 7 typically)
        best_result = max(results, key=lambda x: x['mean_alignment_score'])
        rect = Rectangle((best_result['head']-0.5, best_result['layer']-0.5), 
                        1, 1, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    
    # Shared colorbar
    fig.colorbar(im, ax=axes, label='Alignment Score', fraction=0.046, pad=0.04)
    
    plt.suptitle('Figure 1: Attention-AST Alignment by Layer and Head (Multi-Model)', 
                fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_alignment_heatmap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_alignment_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Created Figure 1: Alignment heatmap")

def plot_layer_wise_performance(alignment_data, probing_data, induction_data, output_dir):
    """Figure 2: Layer-wise Performance Across RQ1-3 (Multi-Model)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    models = ['unixcoder', 'codebert']
    colors = {'unixcoder': '#1f77b4', 'codebert': '#ff7f0e'}
    
    for row, model in enumerate(models):
        # RQ1: Alignment
        task_key = f'{model}_code_to_text'
        if task_key in alignment_data:
            alignment_results = alignment_data[task_key]['all_results']
            layer_alignment = {}
            for result in alignment_results:
                layer = result['layer']
                score = result['mean_alignment_score']
                if layer not in layer_alignment:
                    layer_alignment[layer] = []
                layer_alignment[layer].append(score)
            
            alignment_by_layer = [np.mean(layer_alignment.get(i, [0])) for i in range(13)]
            
            axes[row, 0].plot(range(13), alignment_by_layer, marker='o', 
                            linewidth=2, markersize=6, color=colors[model])
            axes[row, 0].axvline(7, color='red', linestyle='--', alpha=0.5)
            axes[row, 0].set_xlabel('Layer')
            axes[row, 0].set_ylabel('Mean Alignment Score')
            axes[row, 0].set_title(f'{model.upper()}: Attention-AST Alignment')
            axes[row, 0].grid(True, alpha=0.3)
        
        # RQ2: Probing
        if task_key in probing_data:
            probing_results = probing_data[task_key]['layer_summary']
            probing_by_layer = [probing_results.get(f'layer_{i}', 0) for i in range(13)]
            
            axes[row, 1].plot(range(13), probing_by_layer, marker='s', 
                            linewidth=2, markersize=6, color=colors[model])
            axes[row, 1].axvline(7, color='red', linestyle='--', alpha=0.5)
            axes[row, 1].set_xlabel('Layer')
            axes[row, 1].set_ylabel('Spearman ρ')
            axes[row, 1].set_title(f'{model.upper()}: Structural Probing')
            axes[row, 1].grid(True, alpha=0.3)
        
        # RQ3: Tree Induction
        if task_key in induction_data:
            induction_results = induction_data[task_key]['layer_summary']
            induction_by_layer = [induction_results.get(f'layer_{i}', 0) for i in range(13)]
            
            axes[row, 2].plot(range(13), induction_by_layer, marker='^', 
                            linewidth=2, markersize=6, color=colors[model])
            axes[row, 2].axvline(1, color='blue', linestyle='--', alpha=0.5)
            axes[row, 2].set_xlabel('Layer')
            axes[row, 2].set_ylabel('F1 Score')
            axes[row, 2].set_title(f'{model.upper()}: Tree Induction')
            axes[row, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 2: Layer-wise Performance Across Research Questions (Multi-Model)', 
                fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_layer_performance.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_layer_performance.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Created Figure 2: Layer-wise performance")

def plot_construct_comparison(construct_data, output_dir):
    """Figure 3: Go Construct Distribution & Effect Sizes (NEW)."""
    if construct_data is None:
        print("  ⚠ Skipping Figure 3: No construct data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    constructs = ['error_patterns', 'goroutines', 'channels', 'defer', 
                  'interfaces', 'type_assertions']
    construct_labels = ['Error\nPatterns', 'Goroutines', 'Channels', 
                       'Defer', 'Interfaces', 'Type\nAssertions']
    
    # Top left: Prevalence comparison
    task_key = 'unixcoder_code_to_text'
    if task_key in construct_data:
        stats = construct_data[task_key]['prevalence_summary']
        percentages = [stats[c]['percentage'] for c in constructs if c in stats]
        valid_labels = [construct_labels[i] for i, c in enumerate(constructs) if c in stats]
        
        axes[0, 0].barh(range(len(percentages)), percentages, color='steelblue', alpha=0.8)
        axes[0, 0].set_yticks(range(len(percentages)))
        axes[0, 0].set_yticklabels(valid_labels)
        axes[0, 0].set_xlabel('Percentage of Samples (%)')
        axes[0, 0].set_title('Construct Prevalence')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Top right: Effect sizes (Cohen's d) for attention
    unixcoder_effects = []
    codebert_effects = []
    construct_names = []
    
    for construct in constructs:
        if task_key in construct_data and construct in construct_data[task_key]['construct_analyses']:
            analysis = construct_data[task_key]['construct_analyses'][construct]
            if 'attention_analysis' in analysis and analysis['attention_analysis']:
                # Get Layer 7 effect
                layer7_results = [r for r in analysis['attention_analysis'] if r['layer'] == 7]
                if layer7_results and 'entropy' in layer7_results[0]:
                    unixcoder_effects.append(abs(layer7_results[0]['entropy']['cohen_d']))
                    construct_names.append(construct_labels[constructs.index(construct)])
        
        codebert_task = 'codebert_code_to_text'
        if codebert_task in construct_data and construct in construct_data[codebert_task]['construct_analyses']:
            analysis = construct_data[codebert_task]['construct_analyses'][construct]
            if 'attention_analysis' in analysis and analysis['attention_analysis']:
                layer7_results = [r for r in analysis['attention_analysis'] if r['layer'] == 7]
                if layer7_results and 'entropy' in layer7_results[0]:
                    codebert_effects.append(abs(layer7_results[0]['entropy']['cohen_d']))
    
    if unixcoder_effects and codebert_effects:
        x = np.arange(len(construct_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, unixcoder_effects, width, label='UniXcoder', color='#1f77b4')
        axes[0, 1].bar(x + width/2, codebert_effects, width, label='CodeBERT', color='#ff7f0e')
        axes[0, 1].set_ylabel("Cohen's d (Effect Size)")
        axes[0, 1].set_title('Attention Effect Sizes (Layer 7)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(construct_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
        axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Medium')
        axes[0, 1].axhline(0.8, color='gray', linestyle='--', alpha=0.9, label='Large')
    
    # Bottom: Significant layers per construct
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    plt.suptitle('Figure 3: Go Construct Analysis (NOVEL)', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_construct_comparison.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_construct_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Created Figure 3: Construct comparison")

def plot_cka_heatmap(cross_model_data, output_dir):
    """Figure 4: CKA Similarity Heatmap (NEW)."""
    if cross_model_data is None:
        print("  ⚠ Skipping Figure 4: No cross-model data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    tasks = [
        ('code_to_text', 'Code-to-Text', axes[0]),
        ('code_to_code', 'Code-to-Code', axes[1])
    ]
    
    for task_key, title, ax in tasks:
        if task_key not in cross_model_data:
            continue
        
        cka_scores = cross_model_data[task_key]['cka_scores']
        layers = sorted([int(k.split('_')[1]) for k in cka_scores.keys()])
        scores = [cka_scores[f'layer_{i}'] for i in layers]
        
        ax.plot(layers, scores, marker='o', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Layer')
        ax.set_ylabel('CKA Similarity')
        ax.set_title(f'{title}\n(UniXcoder vs CodeBERT)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add interpretation bands
        ax.axhspan(0.8, 1.0, alpha=0.1, color='green', label='Very Similar')
        ax.axhspan(0.5, 0.8, alpha=0.1, color='yellow', label='Moderately Similar')
        ax.axhspan(0, 0.5, alpha=0.1, color='red', label='Different')
    
    plt.suptitle('Figure 4: Representational Similarity (CKA) - UniXcoder vs CodeBERT', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_cka_similarity.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_cka_similarity.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Created Figure 4: CKA similarity")

def plot_model_agreement(cross_model_data, output_dir):
    """Figure 5: Model Agreement Analysis (NEW)."""
    if cross_model_data is None:
        print("  ⚠ Skipping Figure 5: No cross-model data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    task_key = 'code_to_text'
    if task_key not in cross_model_data:
        return
    
    agreement_data = cross_model_data[task_key]['model_agreement']
    
    constructs = list(agreement_data.keys())
    agreement_rates = [agreement_data[c]['agreement_rate'] for c in constructs]
    accuracy_when_agree = [agreement_data[c]['accuracy_when_agree'] for c in constructs]
    
    # Left: Agreement rates
    axes[0].barh(range(len(constructs)), agreement_rates, color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(len(constructs)))
    axes[0].set_yticklabels(constructs)
    axes[0].set_xlabel('Agreement Rate')
    axes[0].set_title('How Often Models Agree')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_xlim(0, 1)
    
    # Right: Accuracy when in agreement
    axes[1].barh(range(len(constructs)), accuracy_when_agree, color='green', alpha=0.8)
    axes[1].set_yticks(range(len(constructs)))
    axes[1].set_yticklabels(constructs)
    axes[1].set_xlabel('Accuracy When Models Agree')
    axes[1].set_title('Joint Accuracy')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xlim(0, 1)
    
    plt.suptitle('Figure 5: Model Agreement Analysis - UniXcoder vs CodeBERT', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_model_agreement.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_model_agreement.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Created Figure 5: Model agreement")

def main():
    """Generate all visualizations."""
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS (FULL)")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading results...")
    try:
        alignment_data = load_alignment_results()
        probing_data = load_probing_results()
        induction_data = load_tree_induction_results()
        construct_data = load_construct_results()
        cross_model_data = load_cross_model_results()
        print("  ✓ Results loaded")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        print("  Make sure all analysis scripts have been run first!")
        return
    
    print("\nGenerating figures...")
    
    # Original figures
    plot_alignment_heatmap(alignment_data, OUTPUT_DIR)
    plot_layer_wise_performance(alignment_data, probing_data, induction_data, OUTPUT_DIR)
    
    # NEW figures
    plot_construct_comparison(construct_data, OUTPUT_DIR)
    plot_cka_heatmap(cross_model_data, OUTPUT_DIR)
    plot_model_agreement(cross_model_data, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("✓ VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • fig1_alignment_heatmap.png/pdf - Multi-model attention-AST alignment")
    print("  • fig2_layer_performance.png/pdf - Multi-model layer-wise performance")
    print("  • fig3_construct_comparison.png/pdf - Construct analysis (NOVEL)")
    print("  • fig4_cka_similarity.png/pdf - Cross-model CKA (NOVEL)")
    print("  • fig5_model_agreement.png/pdf - Model agreement (NOVEL)")
    print("\nAll figures are publication-ready (300 DPI, PDF + PNG)")
    print("\n")

if __name__ == "__main__":
    main()
