# Go Code Understanding: Expanded Deep Learning Analysis

This repository contains the expanded analysis for the Foundations of Deep Learning bonus project, building on the scientific writing paper.

## 📋 Overview

**Original Project**: Structural analysis of Go code understanding in UniXcoder  
**Expanded Project**: Multi-model comparison (UniXcoder vs CodeBERT) with construct-level analysis

### Research Questions (Expanded)

**RQ1 (Prevalence)**: What is the distribution of Go-specific constructs in real-world code?

**RQ2 (Attention Patterns)**: Do attention patterns differ for samples with vs without specific constructs? How does this compare across models?

**RQ3 (Representation Encoding)**: Are Go constructs encoded distinctly in internal representations? How does AST-augmented pre-training (UniXcoder) compare to code-only (CodeBERT)?

**RQ4 (Cross-Model)**: Where do models agree/disagree on construct encoding?

## 🚀 Running on Google Colab

### Step 1: Open Colab Notebook

1. Upload `colab_setup.ipynb` to Google Colab
2. Make sure you're using a **T4 GPU** runtime:
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4)

### Step 2: Expected Runtime

| Task | Time (T4 GPU) | Notes |
|------|---------------|-------|
| Download data | 5-10 min | One-time setup |
| Parse ASTs | 30-45 min | 16,244 samples |
| Extract UniXcoder | 2-3 hours | Full dataset |
| Extract CodeBERT | 2-3 hours | Full dataset |
| All analyses | 1-2 hours | RQ1-4 complete |
| **Total** | **~8-10 hours** | Can run overnight |

### Step 3: Run Cells Sequentially

The Colab notebook is organized as follows:

```python
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Install dependencies
!pip install transformers datasets tree-sitter...

# Cell 3: Clone repository
!git clone https://github.com/amiyilade/go-eval.git

# Cell 4: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 5-8: Data pipeline
!python scripts/download_coir_go.py
!python scripts/organise_coir_go_full.py
!python scripts/parse_go_asts_full.py

# Cell 9-10: Model extraction
!python scripts/extract_model_outputs_full.py --model unixcoder
!python scripts/extract_model_outputs_full.py --model codebert

# Cell 11-15: Analysis
!python scripts/analyze_attention_ast_full.py
!python scripts/structural_probing_full.py
!python scripts/tree_induction_full.py
!python scripts/construct_analysis.py
!python scripts/cross_model_analysis.py

# Cell 16: Generate visualizations
!python scripts/visualizations_full.py

# Cell 17: Save to Drive
!cp -r results/* /content/drive/MyDrive/go_analysis_results/
```

### Step 4: Monitor Progress

The scripts provide progress bars and updates:

```
Processing: full_code_to_text.jsonl (FULL DATASET, BATCHED)
Total samples: 8,122
Batch size: 500
Number of batches: 17

  Processing batch 1 (samples 1 to 500)...
  ✓ Successfully parsed: 500/500 code fields
```

## 📊 Output Structure

After completion, you'll have:

```
results/
├── model_outputs/
│   ├── unixcoder/
│   │   ├── code_to_text_full_unixcoder_features.jsonl
│   │   ├── code_to_code_full_unixcoder_features.jsonl
│   │   └── summaries...
│   └── codebert/
│       ├── code_to_text_full_codebert_features.jsonl
│       └── code_to_code_full_codebert_features.jsonl
│
├── ast_alignment/
│   ├── all_alignment_results.json
│   └── per-model results...
│
├── structural_probing/
│   ├── all_probing_results.json
│   └── per-model results...
│
├── tree_induction/
│   ├── all_tree_induction_results.json
│   └── per-model results...
│
├── construct_analysis/  # NEW!
│   ├── all_construct_analysis.json
│   ├── unixcoder_code_to_text_construct_analysis.json
│   ├── codebert_code_to_text_construct_analysis.json
│   └── ...
│
├── cross_model_analysis/  # NEW!
│   ├── all_cross_model_analysis.json
│   ├── code_to_text_cross_model_analysis.json
│   ├── code_to_code_cross_model_analysis.json
│   └── visualizations/
│       ├── error_patterns_layer7_visualization.png
│       ├── goroutines_layer7_visualization.png
│       └── ...
│
└── visualizations/
    ├── fig1_alignment_heatmap.png
    ├── fig2_layer_performance.png
    ├── fig3_construct_comparison.png  # NEW!
    ├── fig4_cka_heatmap.png  # NEW!
    └── fig5_model_agreement.png  # NEW!
```

## 📈 Key Results Format

### Construct Analysis Results

```json
{
  "model_name": "unixcoder_code_to_text",
  "prevalence_summary": {
    "error_patterns": {
      "samples_with_construct": 6234,
      "percentage": 76.7,
      "total_occurrences": 12456
    }
  },
  "construct_analyses": {
    "error_patterns": {
      "attention_analysis": [
        {
          "layer": 7,
          "head": 7,
          "entropy": {
            "mean_with": 2.45,
            "mean_without": 2.12,
            "p_value": 0.001,
            "cohen_d": 0.42,
            "significant": true
          }
        }
      ],
      "embedding_analysis": [...],
      "significant_attention_layers": [4, 7, 10],
      "significant_embedding_layers": [7]
    }
  }
}
```

### Cross-Model Analysis Results

```json
{
  "cka_scores": {
    "layer_0": 0.95,
    "layer_7": 0.73,
    "layer_12": 0.68
  },
  "classification_accuracy": {
    "error_patterns": {
      "unixcoder": {
        "layer_7": {
          "mean_accuracy": 0.78,
          "std_accuracy": 0.03
        }
      },
      "codebert": {
        "layer_7": {
          "mean_accuracy": 0.72,
          "std_accuracy": 0.04
        }
      }
    }
  },
  "model_agreement": {
    "error_patterns": {
      "agreement_rate": 0.82,
      "accuracy_when_agree": 0.89
    }
  }
}
```

## 🔍 Key Differences from Scientific Writing Paper

| Aspect | Original (Sci Writing) | Expanded (Deep Learning) |
|--------|----------------------|--------------------------|
| **Dataset Size** | 100 samples | 16,244 samples (Full COIR) |
| **Models** | UniXcoder only | UniXcoder + CodeBERT |
| **Constructs** | Distribution only | Statistical tests (t-tests, Cohen's d) |
| **Cross-Model** | N/A | CKA, agreement analysis |
| **Visualization** | Basic heatmaps | PCA/t-SNE, multi-model comparison |
| **Runtime** | 1-2 hours (laptop) | 8-10 hours (Colab T4) |

## 🎯 Expected Contributions

1. **Full Dataset Analysis**: First complete structural analysis on entire COIR Go subset (16,244 samples)

2. **Multi-Model Comparison**: Direct comparison of AST-augmented (UniXcoder) vs code-only (CodeBERT) pre-training strategies

3. **Construct-Level Insights**: Statistical evidence for which Go constructs are encoded differently between architectures

4. **Architectural Insights**: CKA analysis reveals where models converge/diverge in representation space

## 📝 Interpreting Results

### Construct Analysis

**High Cohen's d (>0.5)**: Strong effect - construct substantially changes attention/embedding patterns

**Significant p-value (<0.05)**: Statistically reliable difference between samples with/without construct

**Multiple significant layers**: Construct encoded across multiple levels of abstraction

### Cross-Model Analysis

**High CKA (>0.8)**: Models learn similar representations despite different architectures

**Low CKA (<0.5)**: Architectural differences lead to different encoding strategies

**High agreement + high accuracy**: Both models reliably identify construct presence

**High agreement + low accuracy**: Both models consistently make same mistakes

## 🛠 Troubleshooting

### GPU Out of Memory

```python
# Reduce batch size in extract_model_outputs_full.py
BATCH_SIZE = 4  # Instead of 8
```

### Slow Progress

```python
# Process subset first to verify pipeline
# In organise_coir_go_full.py, add:
if len(code_to_text_data) > 1000:
    code_to_text_data = code_to_text_data[:1000]
```

### Missing Files

```bash
# Verify data downloaded
ls data/raw/codesearchnet/
ls data/raw/codesearchnet-ccr/

# Re-run download if needed
python scripts/download_coir_go.py
```

## 📚 Citation

If you use this code:

```bibtex
@misc{onesi2025go,
  title={Beyond Python: Structural Analysis of Go Code Understanding},
  author={Onesi, Onenamiyi Oluwademilade},
  year={2025},
  howpublished={Foundations of Deep Learning Project}
}
```

## 🤝 Acknowledgments

- Original methodology: Wan et al. (2022) - "What do they capture?"
- COIR benchmark: Li et al. (2025)
- Models: Microsoft Research (UniXcoder, CodeBERT)

## 📧 Contact

For questions: o.onesi@stud.unibas.ch

---

**Good luck with your expanded analysis!** 🚀
