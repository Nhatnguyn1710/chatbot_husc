"""
Visualization Module — Vẽ biểu đồ & ma trận nhầm lẫn cho HUSC RAG Chatbot.

"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (lưu file, không cần GUI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Vietnamese font fallback
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
RESULT_JSON_FILES = {
    "retrieval": "truy_xuat_ket_qua.json",
    "generation": "sinh_ket_qua.json",
    "citation": "trich_dan_ket_qua.json",
    "intent": "y_dinh_ket_qua.json",
    "performance": "hieu_nang_ket_qua.json",
}
CHART_FILES = {
    "retrieval_by_k": "truy_xuat_chi_so_theo_k.png",
    "retrieval_by_category": "truy_xuat_theo_nhom.png",
    "retrieval_mrr_by_query": "truy_xuat_mrr_tung_cau_hoi.png",
    "generation_overall": "sinh_tong_quan.png",
    "generation_by_category": "sinh_theo_nhom.png",
    "citation_metrics": "trich_dan_chi_so.png",
    "intent_confusion_matrix": "y_dinh_ma_tran_nham_lan.png",
    "performance_latency": "hieu_nang_do_tre.png",
}


def _ensure_dirs():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def _load_json(filename: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# 1. RETRIEVAL QUALITY CHARTS
# =========================================================================

def plot_retrieval_metrics(data: dict):
    """Biểu đồ Retrieval: Precision/Recall/NDCG/HitRate theo K + MRR."""
    summary = data.get("summary", {})
    if not summary:
        return

    k_values = [3, 5, 10]
    metrics = ["precision", "recall", "ndcg", "hit_rate"]
    labels = ["Precision@K", "Recall@K", "NDCG@K", "Hit Rate@K"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    # --- Chart 1: Grouped bar chart - Metrics vs K ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(k_values))
    width = 0.2

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [summary.get(f"{metric}@{k}", 0) for k in k_values]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    mrr = summary.get("mrr", 0)
    ax.axhline(y=mrr, color="#9C27B0", linestyle="--", linewidth=1.5, label=f"MRR = {mrr:.3f}")

    ax.set_xlabel("K (Top-K)")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality Metrics theo K")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(CHARTS_DIR, CHART_FILES["retrieval_by_k"])
    fig.savefig(path)
    plt.close(fig)
    print(f"  [1/7] {path}")

    # --- Chart 2: Per-category retrieval performance ---
    cat_data = data.get("category_metrics", {})
    if cat_data:
        categories = sorted(cat_data.keys())
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.22

        cat_mrr = [cat_data[c].get("mrr", 0) for c in categories]
        cat_p5 = [cat_data[c].get("per_k", {}).get(5, cat_data[c].get("per_k", {}).get("5", {})).get("precision", 0) for c in categories]
        cat_r5 = [cat_data[c].get("per_k", {}).get(5, cat_data[c].get("per_k", {}).get("5", {})).get("recall", 0) for c in categories]
        cat_hr5 = [cat_data[c].get("per_k", {}).get(5, cat_data[c].get("per_k", {}).get("5", {})).get("hit_rate", 0) for c in categories]

        ax.bar(x - 1.5*width, cat_mrr, width, label="MRR", color="#9C27B0")
        ax.bar(x - 0.5*width, cat_p5, width, label="P@5", color="#2196F3")
        ax.bar(x + 0.5*width, cat_r5, width, label="R@5", color="#4CAF50")
        ax.bar(x + 1.5*width, cat_hr5, width, label="HR@5", color="#E91E63")

        counts = [cat_data[c].get("count", 0) for c in categories]
        cat_labels = [f"{c}\n(n={n})" for c, n in zip(categories, counts)]

        ax.set_xlabel("Category")
        ax.set_ylabel("Score")
        ax.set_title("Retrieval Quality theo Category")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        path = os.path.join(CHARTS_DIR, CHART_FILES["retrieval_by_category"])
        fig.savefig(path)
        plt.close(fig)
        print(f"  [2/7] {path}")

    # --- Chart 3: Per-query detail (hit/miss) ---
    details = data.get("details", [])
    if details:
        fig, ax = plt.subplots(figsize=(12, 5))
        queries = [d["query"][:30] + "..." if len(d["query"]) > 30 else d["query"] for d in details]
        mrr_vals = [d.get("mrr", 0) for d in details]
        bar_colors = ["#4CAF50" if m > 0 else "#F44336" for m in mrr_vals]

        bars = ax.barh(range(len(queries)), mrr_vals, color=bar_colors, edgecolor="white")
        ax.set_yticks(range(len(queries)))
        ax.set_yticklabels(queries, fontsize=8)
        ax.set_xlabel("MRR (1.0 = top-1 hit)")
        ax.set_title("Retrieval MRR per Query (xanh=hit, do=miss)")
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        path = os.path.join(CHARTS_DIR, CHART_FILES["retrieval_mrr_by_query"])
        fig.savefig(path)
        plt.close(fig)
        print(f"  [3/7] {path}")


# =========================================================================
# 2. GENERATION / ANSWER ACCURACY CHARTS
# =========================================================================

def plot_generation_metrics(data: dict):
    """Biểu đồ Generation: ROUGE, BLEU, Key Fact Coverage."""
    summary = data.get("summary", {})
    if not summary:
        return

    # --- Chart 4: Overall generation metrics ---
    metrics = {
        "ROUGE-1": summary.get("rouge1_f1", 0),
        "ROUGE-2": summary.get("rouge2_f1", 0),
        "ROUGE-L": summary.get("rougeL_f1", 0),
        "BLEU": summary.get("bleu", 0),
        "Key Facts": summary.get("key_fact_coverage", 0),
    }
    bert = summary.get("bertscore_f1")
    if bert:
        metrics["BERTScore"] = bert

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#2196F3", "#1976D2", "#0D47A1", "#FF9800", "#4CAF50", "#9C27B0"][:len(names)]

    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Score (F1)")
    ax.set_title("Answer Accuracy — Generation Quality Metrics")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(CHARTS_DIR, CHART_FILES["generation_overall"])
    fig.savefig(path)
    plt.close(fig)
    print(f"  [4/7] {path}")

    # --- Chart 5: Per-category generation ---
    cat_data = data.get("category_metrics", {})
    if cat_data:
        categories = sorted(cat_data.keys())
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.18

        metrics_list = [
            ("rouge1_f1", "ROUGE-1", "#2196F3"),
            ("rouge2_f1", "ROUGE-2", "#1976D2"),
            ("rougeL_f1", "ROUGE-L", "#0D47A1"),
            ("bleu", "BLEU", "#FF9800"),
            ("key_fact_coverage", "Key Facts", "#4CAF50"),
        ]

        for i, (key, label, color) in enumerate(metrics_list):
            values = [cat_data[c].get(key, 0) for c in categories]
            ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white")

        counts = [cat_data[c].get("count", 0) for c in categories]
        cat_labels = [f"{c}\n(n={n})" for c, n in zip(categories, counts)]

        ax.set_xlabel("Category")
        ax.set_ylabel("Score")
        ax.set_title("Generation Metrics theo Category")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(cat_labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(axis="y", alpha=0.3)

        path = os.path.join(CHARTS_DIR, CHART_FILES["generation_by_category"])
        fig.savefig(path)
        plt.close(fig)
        print(f"  [5/7] {path}")


# =========================================================================
# 3. CITATION & HALLUCINATION CHART
# =========================================================================

def plot_citation_metrics(data: dict):
    """Biểu đồ Citation: Precision, Recall, Clean Rate, Hallucination."""
    summary = data.get("summary", {})
    if not summary:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart
    ax = axes[0]
    metrics = {
        "Citation\nPrecision": summary.get("citation_precision", 0),
        "Citation\nRecall": summary.get("citation_recall", 0),
        "Clean\nRate": summary.get("clean_rate", 0),
    }
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_title("Citation Accuracy")
    ax.grid(axis="y", alpha=0.3)

    # Right: hallucination donut
    ax = axes[1]
    hall_rate = summary.get("hallucination_rate", 0)
    clean_rate = 1 - hall_rate
    sizes = [clean_rate, hall_rate]
    labels = [f"Clean\n{clean_rate:.1%}", f"Halluc.\n{hall_rate:.1%}"]
    colors_pie = ["#4CAF50", "#F44336"]
    wedges, texts = ax.pie(sizes, labels=labels, colors=colors_pie,
                           startangle=90, wedgeprops={"width": 0.4, "edgecolor": "white"})
    for t in texts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title("Hallucination Rate")

    fig.suptitle("Citation & Anti-Hallucination Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(CHARTS_DIR, CHART_FILES["citation_metrics"])
    fig.savefig(path)
    plt.close(fig)
    print(f"  [6/7] {path}")


# =========================================================================
# 4. INTENT CONFUSION MATRIX
# =========================================================================

def plot_intent_confusion_matrix(data: dict):
    """Ma trận nhầm lẫn heatmap cho Intent Classification."""
    confusion = data.get("confusion_matrix", {})
    per_class = data.get("per_class", {})
    summary = data.get("summary", {})
    if not confusion:
        return

    classes = ["small_talk", "out_of_scope", "academic"]
    class_labels = ["Small Talk", "Out of Scope", "Academic"]

    # Build matrix
    matrix = np.array([[confusion.get(a, {}).get(p, 0) for p in classes] for a in classes])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: Confusion matrix heatmap
    ax = axes[0]
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels,
                yticklabels=class_labels, ax=ax, cbar_kws={"shrink": 0.8},
                linewidths=0.5, linecolor="white", annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    accuracy = summary.get("accuracy", 0)
    ax.set_title(f"Confusion Matrix\n(Accuracy: {accuracy:.1%})", fontsize=13)

    # Right: Per-class F1 bar chart
    ax = axes[1]
    f1_scores = [per_class.get(c, {}).get("f1", 0) for c in classes]
    precision_scores = [per_class.get(c, {}).get("precision", 0) for c in classes]
    recall_scores = [per_class.get(c, {}).get("recall", 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, precision_scores, width, label="Precision", color="#2196F3", edgecolor="white")
    ax.bar(x, recall_scores, width, label="Recall", color="#4CAF50", edgecolor="white")
    ax.bar(x + width, f1_scores, width, label="F1", color="#FF9800", edgecolor="white")

    for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
        ax.text(i + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Metrics")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Intent Classification Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(CHARTS_DIR, CHART_FILES["intent_confusion_matrix"])
    fig.savefig(path)
    plt.close(fig)
    print(f"  [7/7] {path}")


# =========================================================================
# 5. PERFORMANCE LATENCY CHART (bonus)
# =========================================================================

def plot_performance_metrics(data: dict):
    """Biểu đồ latency breakdown (bonus)."""
    summary = data.get("summary", {})
    bd = summary.get("latency_breakdown_pct", {})
    if not bd:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pie chart
    ax = axes[0]
    sizes = [bd.get("retrieve", 0), bd.get("rerank", 0), bd.get("generate", 0)]
    labels = [f"Retrieve\n{sizes[0]:.1f}%", f"Rerank\n{sizes[1]:.1f}%", f"Generate\n{sizes[2]:.1f}%"]
    colors = ["#2196F3", "#FF9800", "#E91E63"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    e2e = summary.get("e2e_latency", {})
    ax.set_title(f"Latency Breakdown\n(Mean E2E: {e2e.get('mean', 0):.0f}ms)")

    # Right: per-query latency bar
    details = data.get("details", [])
    if details:
        ax = axes[1]
        queries = [d["query"][:25] + "..." if len(d["query"]) > 25 else d["query"] for d in details]
        e2e_vals = [d.get("e2e_ms", 0) for d in details]
        ret_vals = [d.get("retrieve_ms", 0) for d in details]
        rer_vals = [d.get("rerank_ms", 0) for d in details]
        gen_vals = [d.get("generate_ms", 0) for d in details]

        y = np.arange(len(queries))
        ax.barh(y, ret_vals, label="Retrieve", color="#2196F3")
        ax.barh(y, rer_vals, left=ret_vals, label="Rerank", color="#FF9800")
        left2 = [r + rr for r, rr in zip(ret_vals, rer_vals)]
        ax.barh(y, gen_vals, left=left2, label="Generate", color="#E91E63")

        ax.set_yticks(y)
        ax.set_yticklabels(queries, fontsize=7)
        ax.set_xlabel("Latency (ms)")
        ax.set_title("Per-query Latency Breakdown")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

    fig.suptitle("System Performance", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(CHARTS_DIR, CHART_FILES["performance_latency"])
    fig.savefig(path)
    plt.close(fig)
    print(f"  [bonus] {path}")


# =========================================================================
# MAIN
# =========================================================================

def generate_all_charts(verbose: bool = True):
    """Đọc JSON results và vẽ tất cả biểu đồ."""
    _ensure_dirs()

    if verbose:
        print("=" * 60)
        print("📊 GENERATING CHARTS")
        print("=" * 60)

    # Retrieval
    ret_filename = RESULT_JSON_FILES["retrieval"]
    ret_data = _load_json(ret_filename)
    if ret_data:
        if verbose:
            print("\n🔍 Retrieval Quality Charts:")
        plot_retrieval_metrics(ret_data)
    else:
        print(f"  ⚠️ {ret_filename} not found — skip")

    # Generation
    gen_filename = RESULT_JSON_FILES["generation"]
    gen_data = _load_json(gen_filename)
    if gen_data:
        if verbose:
            print("\n📝 Generation Quality Charts:")
        plot_generation_metrics(gen_data)
    else:
        print(f"  ⚠️ {gen_filename} not found — skip")

    # Citation
    cit_filename = RESULT_JSON_FILES["citation"]
    cit_data = _load_json(cit_filename)
    if cit_data:
        if verbose:
            print("\n📌 Citation Charts:")
        plot_citation_metrics(cit_data)
    else:
        print(f"  ⚠️ {cit_filename} not found — skip")

    # Intent
    int_filename = RESULT_JSON_FILES["intent"]
    int_data = _load_json(int_filename)
    if int_data:
        if verbose:
            print("\n🎯 Intent Confusion Matrix:")
        plot_intent_confusion_matrix(int_data)
    else:
        print(f"  ⚠️ {int_filename} not found — skip")

    # Performance (bonus)
    perf_filename = RESULT_JSON_FILES["performance"]
    perf_data = _load_json(perf_filename)
    if perf_data:
        if verbose:
            print("\n⚡ Performance Charts:")
        plot_performance_metrics(perf_data)

    if verbose:
        print(f"\n✅ Charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    generate_all_charts()
