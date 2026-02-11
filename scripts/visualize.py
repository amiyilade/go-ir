#!/usr/bin/env python3
"""
Script 9: Visualize
Reads all results JSON files and generates publication-quality figures.

Works with UniXcoder alone.  If CodeBERT results are also present,
comparison panels are added automatically.

Output: results/figures/*.png  +  *.pdf

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --model unixcoder   # force single-model mode
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

RESULTS_DIR = Path("results")
FIG_DIR     = RESULTS_DIR / "figures"
MODELS      = ["unixcoder", "codebert"]
TASKS       = ["code_to_text", "code_to_code"]
TASK_LABELS = {"code_to_text": "Code-to-Text", "code_to_code": "Code-to-Code"}
MODEL_COLORS = {"unixcoder": "#1f77b4", "codebert": "#ff7f0e"}

GO_CONSTRUCTS = [
    "goroutines", "channels", "defer", "error_patterns",
    "select_statements", "interfaces", "type_assertions", "context_usage",
]
CONSTRUCT_LABELS = [
    "Goroutines", "Channels", "Defer", "Error\nPatterns",
    "Select", "Interfaces", "Type\nAssertions", "Context",
]


# ── helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def available_models(task: str) -> list:
    return [m for m in MODELS
            if (RESULTS_DIR / f"rq2_{task}_{m}.json").exists()]


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png / .pdf")


# ── Fig 1: RQ1 — Construct Prevalence ─────────────────────────────────────

def fig_prevalence():
    datasets = {t: load_json(RESULTS_DIR / f"rq1_prevalence_{t}.json") for t in TASKS}
    if not any(datasets.values()):
        print("  ⚠ No RQ1 prevalence data — skipping Fig 1")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, task in zip(axes, TASKS):
        data = datasets.get(task)
        if not data:
            ax.set_visible(False)
            continue
        stats = data["construct_stats"]
        pcts  = [stats.get(c, {}).get("prevalence_pct", 0) for c in GO_CONSTRUCTS]
        bars  = ax.bar(range(len(GO_CONSTRUCTS)), pcts,
                       color=["#c0392b" if c == "error_patterns" else "#2980b9"
                              for c in GO_CONSTRUCTS], alpha=0.85)
        ax.set_xticks(range(len(GO_CONSTRUCTS)))
        ax.set_xticklabels(CONSTRUCT_LABELS, rotation=40, ha="right")
        ax.set_ylabel("Samples containing construct (%)")
        ax.set_title(TASK_LABELS[task])
        ax.grid(axis="y", alpha=0.3)
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 1 · Go Construct Prevalence in Real-World Code (RQ1)", y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig1_prevalence")


# ── Fig 2: RQ2 — Alignment Heatmap ────────────────────────────────────────

def get_mean_alignment(h):
    return h.get("mean_alignment", h.get("mean"))

def get_head_id(h):
    return h.get("head_id", h.get("head"))

def fig_alignment_heatmap():
    for model in MODELS:
        for task in TASKS:
            data = load_json(RESULTS_DIR / f"rq2_{task}_{model}.json")
            if not data:
                continue

            layer_summary = data["part_a_attention_ast_alignment"]["layer_summary"]

            # handle JSON string keys safely
            layers = sorted(layer_summary, key=lambda x: int(x.replace("layer_", "")))
            n_lay = len(layers)

            mat = np.zeros((n_lay, 1))  # single column = per-layer alignment

            for i, layer in enumerate(layers):
                mat[i, 0] = get_mean_alignment(layer_summary[layer])

            fig, ax = plt.subplots(figsize=(3, 5))
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=mat.max())

            ax.set_xticks([0])
            ax.set_xticklabels(["Mean"])
            ax.set_yticks(range(n_lay))
            ax.set_yticklabels([f"L{l.replace('layer_', '')}" for l in layers])

            ax.set_xlabel("Alignment")
            ax.set_ylabel("Layer")
            ax.set_title(f"Attention-AST Alignment — {model} / {TASK_LABELS[task]}")

            plt.colorbar(im, ax=ax, label="Alignment score")
            plt.tight_layout()
            save_fig(fig, f"fig2_alignment_{task}_{model}")


# ── Fig 3: RQ2/RQ3 — Layer-wise Performance ───────────────────────────────

def fig_layerwise():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["RQ2a: Attention-AST Alignment",
              "RQ3a: Structural Probing (ρ)",
              "RQ3b: Tree Induction (F1)"]

    for model in MODELS:
        color = MODEL_COLORS[model]
        for task in TASKS:
            ls = "--" if task == "code_to_code" else "-"
            label = f"{model}/{TASK_LABELS[task]}"

            rq2 = load_json(RESULTS_DIR / f"rq2_{task}_{model}.json")
            rq3 = load_json(RESULTS_DIR / f"rq3_{task}_{model}.json")

            if rq2:
                ls_sum = rq2["part_a_attention_ast_alignment"]["layer_summary"]
                layers = sorted(int(k.split("_")[1]) for k in ls_sum)
                vals   = [ls_sum[f"layer_{l}"]["mean"] for l in layers]
                axes[0].plot(layers, vals, ls, color=color, label=label, lw=1.8)

            if rq3:
                probe_sum = rq3["part_a_structural_probing"]["layer_summary"]
                layers = sorted(int(k.split("_")[1]) for k in probe_sum)
                vals   = [probe_sum[f"layer_{l}"] for l in layers]
                axes[1].plot(layers, vals, ls, color=color, label=label, lw=1.8)

                tree_sum = rq3["part_b_tree_induction"]["layer_summary"]
                layers = sorted(int(k.split("_")[1]) for k in tree_sum)
                vals   = [tree_sum[f"layer_{l}"] for l in layers]
                axes[2].plot(layers, vals, ls, color=color, label=label, lw=1.8)

    ylabels = ["Alignment score", "Spearman ρ", "F1"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Figure 3 · Layer-wise Performance Across RQs", y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig3_layerwise")


# ── Fig 4: RQ4 — Classification Probe F1 per Construct ────────────────────

def fig_probe_f1():
    collected = {}
    for model in MODELS:
        for task in TASKS:
            data = load_json(RESULTS_DIR / f"rq4_{task}_{model}.json")
            if data:
                collected[(model, task)] = data

    if not collected:
        print("  ⚠ No RQ4 data — skipping Fig 4")
        return

    fig, axes = plt.subplots(1, len(collected), figsize=(6 * len(collected), 5))
    if len(collected) == 1:
        axes = [axes]

    for ax, (key, data) in zip(axes, collected.items()):
        model, task = key
        probes = data["part_a_classification_probes"]
        best_f1 = []
        for c in GO_CONSTRUCTS:
            if c in probes and probes[c].get("best_layer"):
                best_f1.append(probes[c]["best_layer"]["auroc"])
            else:
                best_f1.append(0.0)

        colors = ["#c0392b" if c == "error_patterns" else "#27ae60" for c in GO_CONSTRUCTS]
        ax.barh(range(len(GO_CONSTRUCTS)), best_f1, color=colors, alpha=0.85)
        ax.axvline(0.5, color="gray", ls="--", lw=1, label="Chance")
        ax.set_yticks(range(len(GO_CONSTRUCTS)))
        ax.set_yticklabels(CONSTRUCT_LABELS)
        ax.set_xlabel("Best-layer F1")
        ax.set_title(f"{model} / {TASK_LABELS[task]}")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Figure 4 · Classification Probe F1 — Can Embeddings Detect Go Constructs? (RQ4a)",
                 y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig4_probe_f1")


# ── Fig 5: RQ4 — Probe F1 by Layer for Key Constructs ─────────────────────

def fig_probe_layer_curves():
    key_constructs = ["error_patterns", "goroutines", "channels", "defer"]

    for model in MODELS:
        for task in TASKS:
            data = load_json(RESULTS_DIR / f"rq4_{task}_{model}.json")
            if not data:
                continue
            probes = data["part_a_classification_probes"]

            fig, ax = plt.subplots(figsize=(7, 4))
            for c in key_constructs:
                if c not in probes:
                    continue
                rows = probes[c].get("all_layers", [])
                if not rows:
                    continue
                rows_sorted = sorted(rows, key=lambda x: x["layer"])
                layers = [r["layer"] for r in rows_sorted]
                f1s    = [r["auroc"]    for r in rows_sorted]
                ax.plot(layers, f1s, marker="o", lw=1.8, label=c.replace("_", " "))

            ax.axhline(0.5, color="gray", ls="--", lw=1, label="Chance")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cross-val F1")
            ax.set_title(f"Probe F1 by Layer — {model} / {TASK_LABELS[task]}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            save_fig(fig, f"fig5_probe_curves_{task}_{model}")


# ── Fig 6: RQ4 — t-SNE Scatter ────────────────────────────────────────────

def fig_tsne():
    for model in MODELS:
        for task in TASKS:
            data = load_json(RESULTS_DIR / f"rq4_{task}_{model}.json")
            if not data:
                continue
            tsne_data = data.get("part_b_representation", {}).get("tsne", {})
            if not tsne_data:
                continue

            # Visualise layer 7 if available, else first available
            layer_key = "layer_7" if "layer_7" in tsne_data else next(iter(tsne_data), None)
            if not layer_key:
                continue
            layer_constructs = tsne_data[layer_key]
            if not layer_constructs:
                continue

            n_plots = len(layer_constructs)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
            if n_plots == 1:
                axes = [axes]

            for ax, (c, coords) in zip(axes, layer_constructs.items()):
                if not coords:
                    ax.set_visible(False)
                    continue
                x  = np.array(coords["x"])
                y  = np.array(coords["y"])
                lb = np.array(coords["labels"])
                ax.scatter(x[lb == 0], y[lb == 0], s=8,  alpha=0.4,
                           c="#95a5a6", label="absent")
                ax.scatter(x[lb == 1], y[lb == 1], s=14, alpha=0.7,
                           c="#e74c3c", label="present")
                ax.set_title(c.replace("_", " "))
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                ax.legend(fontsize=8, markerscale=2)
                ax.axis("off")

            fig.suptitle(
                f"Figure 6 · t-SNE ({layer_key}) — "
                f"{model} / {TASK_LABELS[task]}  (RQ4b)", y=1.02)
            plt.tight_layout()
            save_fig(fig, f"fig6_tsne_{task}_{model}")


# ── Fig 7 (optional): Cross-model comparison ──────────────────────────────

def get_max_alignment(h):
    return h.get("max_alignment", h.get("max"))

def fig_model_comparison():
    """Only rendered if both UniXcoder and CodeBERT results exist."""
    has_both = all(
        (RESULTS_DIR / f"rq2_{task}_{m}.json").exists()
        for m in MODELS for task in TASKS
    )
    if not has_both:
        print("  ⚠ CodeBERT results not found — skipping cross-model comparison Fig 7")
        return

    metrics = {"alignment": {}, "probing": {}, "tree_f1": {}}

    for model in MODELS:
        for task in TASKS:
            rq2 = load_json(RESULTS_DIR / f"rq2_{task}_{model}.json")
            rq3 = load_json(RESULTS_DIR / f"rq3_{task}_{model}.json")
            key = f"{model}/{TASK_LABELS[task]}"

            if rq2:
                metrics["alignment"][key] = get_max_alignment(rq2["part_a_attention_ast_alignment"]["overall"])
            if rq3:
                bp = rq3["part_a_structural_probing"].get("best_layer", {})
                bt = rq3["part_b_tree_induction"].get("best_layer", {})
                metrics["probing"][key]  = bp.get("spearman", 0)
                metrics["tree_f1"][key]  = bt.get("mean_f1", 0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (title, m_dict) in zip(axes,
        [("Max Alignment", metrics["alignment"]),
         ("Best Probing ρ", metrics["probing"]),
         ("Best Tree F1", metrics["tree_f1"])]):

        keys = list(m_dict.keys())
        vals = list(m_dict.values())
        colors = [MODEL_COLORS["unixcoder"] if "unixcoder" in k else MODEL_COLORS["codebert"]
                  for k in keys]
        ax.barh(keys, vals, color=colors, alpha=0.85)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Figure 7 · UniXcoder vs CodeBERT Comparison", y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig7_model_comparison")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS,
                        help="Force single-model mode (default: auto-detect)")
    args = parser.parse_args()

    print("=" * 60)
    print("SCRIPT 9: VISUALIZE")
    print("=" * 60)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures …")
    fig_prevalence()
    fig_alignment_heatmap()
    fig_layerwise()
    fig_probe_f1()
    fig_probe_layer_curves()
    fig_tsne()
    fig_model_comparison()

    figs = list(FIG_DIR.glob("*.png"))
    print(f"\n{'='*60}")
    print(f"✓ DONE — {len(figs)} figures in {FIG_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
