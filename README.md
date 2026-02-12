# Go Code Understanding: Expanded Deep Learning Analysis

Multi-model comparison (UniXcoder vs CodeBERT) with construct-level analysis across four research questions.

## Research Questions

**RQ1 (Prevalence)**: What is the distribution of Go-specific constructs in real-world code?

**RQ2 (Attention Patterns)**: Do attention patterns differ for samples with vs without specific constructs? How does this compare across models?

**RQ3 (Representation Encoding)**: Are Go constructs encoded distinctly in internal representations? How does AST-augmented pre-training (UniXcoder) compare to code-only (CodeBERT)?

**RQ4 (Supervision Effects)**: How does AST supervision alter layer-wise encoding trajectories and construct accessibility across models?

## Running on Google Colab

### Prerequisites
- A100 GPU runtime (or better)
- ~10-16 hours for full pipeline

### Expected Runtime per Task

| Task | Time (A100 GPU) |
|------|---------------|
| Download & parse data | 40-50 min |
| Extract UniXcoder features | 2-3 hours |
| Extract CodeBERT features | 2-3 hours |
| RQ1-4 analyses | 1-2 hours |
| **Total** | **~8-10 hours** |

### Pipeline Overview

The analysis pipeline processes the full Go subset of CoIR (16,244 samples). Certain visualisation steps operate on stratified subsets (e.g., 2,000 samples) for computational efficiency.

1. Data download and AST parsing
2. UniXcoder feature extraction (code-to-text & code-to-code)
3. CodeBERT feature extraction (code-to-text & code-to-code)
4. RQ1: Construct prevalence analysis
5. RQ2: Attention pattern analysis (per-model)
6. RQ3: Structural representation probing
7. RQ4: Cross-model agreement analysis
8. Visualization generation
9. Results aggregation


# go-ir/ — Full Directory Structure After Running All Scripts

go-ir/
│
├── data/
│   ├── code-to-text/
│   │   ├── full_code_to_text.jsonl                        
│   │   ├── full_code_to_text_constructs.jsonl             
│   │   ├── stratified_2k_code_to_text.jsonl               
│   │   └── stratified_2k_code_to_text_with_asts.jsonl    
│   │
│   └── code-to-code/
│       ├── full_code_to_code.jsonl                        
│       ├── full_code_to_code_constructs.jsonl             
│       ├── stratified_2k_code_to_code.jsonl              
│       └── stratified_2k_code_to_code_with_asts.jsonl    
│
├── features/                                              
│   ├── code_to_text_unixcoder.h5
│   ├── code_to_text_codebert.h5
│   ├── code_to_code_unixcoder.h5
│   └── code_to_code_codebert.h5
│
├── results/                                               
│   ├── rq1_prevalence.json                                
│   │
│   ├── rq2_code_to_text_unixcoder.json                    
│   ├── rq2_code_to_text_codebert.json
│   ├── rq2_code_to_code_unixcoder.json
│   ├── rq2_code_to_code_codebert.json
│   │
│   ├── rq3_code_to_text_unixcoder.json                    
│   ├── rq3_code_to_text_codebert.json
│   ├── rq3_code_to_code_unixcoder.json
│   ├── rq3_code_to_code_codebert.json
│   │
│   ├── rq4_code_to_text_unixcoder.json                   
│   ├── rq4_code_to_text_codebert.json
│   ├── rq4_code_to_code_unixcoder.json                    # produced but excluded from paper
│   └── rq4_code_to_code_codebert.json                    # produced but excluded from paper
│
├── figures/                                               
│   │
│   ├── fig1_construct_prevalence.png / .pdf               
│   ├── fig2_attention_heatmap_unixcoder_code_to_text.png / .pdf
│   ├── fig2_attention_heatmap_unixcoder_code_to_code.png / .pdf
│   ├── fig2_attention_heatmap_codebert_code_to_text.png / .pdf
│   ├── fig2_attention_heatmap_codebert_code_to_code.png / .pdf
│   ├── fig3_layer_performance.png / .pdf
│   ├── fig4_construct_probing_auroc.png / .pdf
│   ├── fig5_pca_centroid_distances.png / .pdf
│   ├── fig6_model_comparison.png / .pdf
│   ├── fig7_complete_synthesis.png / .pdf
│   │
│   ├── rq4_tsne_code_to_text_unixcoder_goroutines.png     # Script 8 — one per construct per model
│   ├── rq4_tsne_code_to_text_unixcoder_channels.png
│   ├── rq4_tsne_code_to_text_unixcoder_defer.png
│   ├── rq4_tsne_code_to_text_unixcoder_error_patterns.png
│   ├── rq4_tsne_code_to_text_unixcoder_select_statements.png
│   ├── rq4_tsne_code_to_text_unixcoder_interfaces.png
│   ├── rq4_tsne_code_to_text_unixcoder_type_assertions.png
│   ├── rq4_tsne_code_to_text_unixcoder_context_usage.png
│   ├── rq4_tsne_code_to_text_codebert_goroutines.png      # Same set for CodeBERT
│   ├── rq4_tsne_code_to_text_codebert_channels.png
│   ├── ... (6 more)
│   └── rq4_tsne_code_to_text_codebert_context_usage.png
│
├── scripts/
    ├── utils/
    │   ├── __init__.py
    │   └── data_loader.py
    ├── 1_extract_go_constructs.py
    ├── 2_create_stratified_sample.py
    ├── 3_parse_asts.py
    ├── 4_extract_features.py
    ├── 5_rq1_prevalence.py
    ├── 6_rq2_attention.py
    ├── 7_rq3_probing.py
    ├── 8_rq4_constructs.py
    └── 9_visualize.py

## Key Findings

- CodeBERT achieves peak structural probing performance at Layer 0 (embedding layer)
- UniXcoder shows progressive syntactic refinement through middle layers
- 100% of attention heads exhibit position-based patterns in Go code analysis
- Only goroutines and context usage show meaningful classification signal
- Architecturally divergent encoding strategies between code-only vs AST-augmented pre-training

## Citation
```bibtex
@misc{onesi2026go,
  title={Supervision Shapes Representation: A Comparative Study of Construct Encoding in Code Transformers},
  author={Onesi, Onenamiyi Oluwademilade},
  year={2026},
  howpublished={Foundations of Deep Learning Project}
}
```

## Acknowledgments

- Methodology: Wan et al. (2022) - "What do they capture?"
- Models: Microsoft Research (UniXcoder, CodeBERT)

## Contact

o.onesi@stud.unibas.ch