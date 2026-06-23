
from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_core import RAGEngine
from metrics.evaluation_dataset import CITATION_TESTS, GENERATION_TESTS, RETRIEVAL_TESTS


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ============================================================
# Retrieval metrics
# ============================================================

def _is_relevant(result: dict, test_case: dict) -> bool:
    expected_articles = test_case.get("expected_articles", [])
    expected_sources = test_case.get("expected_sources", [])

    if expected_articles:
        text = result.get("text", "")
        rec_article = str(result.get("article", ""))
        for art in expected_articles:
            if rec_article == art:
                return True
            if re.search(rf"(?:^|\n)\s*(?:Dieu|Điều)\s+{art}[\s:.]", text, re.IGNORECASE):
                return True
        return False

    if expected_sources:
        source = str(result.get("source", "")).lower()
        for src_type in expected_sources:
            if src_type == "csv" and source.endswith(".csv"):
                return True
            if src_type == "pdf" and source.endswith(".pdf"):
                return True
    return False


def precision_at_k(results: list[dict], test_case: dict, k: int = 5) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if _is_relevant(r, test_case))
    return relevant / len(top_k)


def recall_at_k(results: list[dict], test_case: dict, k: int = 5, total_relevant: int = 1) -> float:
    top_k = results[:k]
    if total_relevant == 0:
        return 1.0
    relevant = sum(1 for r in top_k if _is_relevant(r, test_case))
    return min(1.0, relevant / total_relevant)


def mrr(results: list[dict], test_case: dict) -> float:
    for i, result in enumerate(results):
        if _is_relevant(result, test_case):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(results: list[dict], test_case: dict, k: int = 5) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0

    dcg = 0.0
    for i, result in enumerate(top_k):
        rel = 1.0 if _is_relevant(result, test_case) else 0.0
        dcg += rel / math.log2(i + 2)

    n_relevant = sum(1 for r in results if _is_relevant(r, test_case))
    ideal_rels = [1.0] * min(n_relevant, k) + [0.0] * max(0, k - n_relevant)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def _hit(results: list[dict], test_case: dict, k: int = 5) -> bool:
    return any(_is_relevant(r, test_case) for r in results[:k])


def evaluate_retrieval(engine: RAGEngine, k_values: list[int] | None = None, verbose: bool = True) -> dict:
    k_values = k_values or [3, 5, 10]
    results_detail = []
    metrics_per_k = {k: {"precision": [], "recall": [], "ndcg": [], "hits": []} for k in k_values}
    mrr_scores = []

    if verbose:
        print("=" * 70)
        print("RETRIEVAL QUALITY METRICS")
        print("=" * 70)

    for i, tc in enumerate(RETRIEVAL_TESTS):
        query = tc["query"]
        if verbose:
            print(f"\n[{i + 1}/{len(RETRIEVAL_TESTS)}] {query}")

        start_t = time.perf_counter()
        context = engine.retrieve(query, top_k=max(k_values) * 2)
        reranked = engine.rerank_results(query, context, top_k=max(k_values))
        reranked = engine.apply_metadata_boost(query, reranked)
        latency = (time.perf_counter() - start_t) * 1000

        mrr_val = mrr(reranked, tc)
        mrr_scores.append(mrr_val)

        tc_result = {
            "query": query,
            "category": tc.get("category", ""),
            "description": tc.get("description", ""),
            "mrr": mrr_val,
            "latency_ms": latency,
            "metrics_per_k": {},
        }

        for k in k_values:
            p = precision_at_k(reranked, tc, k)
            r = recall_at_k(reranked, tc, k)
            n = ndcg_at_k(reranked, tc, k)
            h = 1.0 if _hit(reranked, tc, k) else 0.0

            metrics_per_k[k]["precision"].append(p)
            metrics_per_k[k]["recall"].append(r)
            metrics_per_k[k]["ndcg"].append(n)
            metrics_per_k[k]["hits"].append(h)

            tc_result["metrics_per_k"][k] = {
                "precision": p,
                "recall": r,
                "ndcg": n,
                "hit": h,
            }

        if verbose:
            top1 = reranked[0] if reranked else {}
            art = top1.get("article", "?")
            score = top1.get("final_score", 0)
            print(f"  Top-1: article={art}, score={score:.3f}, MRR={mrr_val:.3f}, latency={latency:.0f}ms")

        results_detail.append(tc_result)

    categories = set(tc.get("category", "") for tc in RETRIEVAL_TESTS)
    category_metrics = {}
    for cat in categories:
        cat_indices = [i for i, tc in enumerate(RETRIEVAL_TESTS) if tc.get("category") == cat]
        if not cat_indices:
            continue
        cat_mrr = [mrr_scores[i] for i in cat_indices]
        cat_data = {}
        for k in k_values:
            cat_data[k] = {
                "precision": _avg([metrics_per_k[k]["precision"][i] for i in cat_indices]),
                "recall": _avg([metrics_per_k[k]["recall"][i] for i in cat_indices]),
                "ndcg": _avg([metrics_per_k[k]["ndcg"][i] for i in cat_indices]),
                "hit_rate": _avg([metrics_per_k[k]["hits"][i] for i in cat_indices]),
            }
        category_metrics[cat] = {"mrr": _avg(cat_mrr), "per_k": cat_data, "count": len(cat_indices)}

    summary = {"mrr": _avg(mrr_scores)}
    for k in k_values:
        summary[f"precision@{k}"] = _avg(metrics_per_k[k]["precision"])
        summary[f"recall@{k}"] = _avg(metrics_per_k[k]["recall"])
        summary[f"ndcg@{k}"] = _avg(metrics_per_k[k]["ndcg"])
        summary[f"hit_rate@{k}"] = _avg(metrics_per_k[k]["hits"])

    return {
        "summary": summary,
        "category_metrics": category_metrics,
        "details": results_detail,
        "test_count": len(RETRIEVAL_TESTS),
    }


# ============================================================
# Generation metrics
# ============================================================

def _tokenize_vi(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(hypothesis: str, reference: str, n: int = 1) -> dict:
    hyp_tokens = _tokenize_vi(hypothesis)
    ref_tokens = _tokenize_vi(reference)

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0, "recall": 0, "f1": 0}

    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    overlap = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams if ng in ref_ngrams)

    precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> dict:
    hyp_tokens = _tokenize_vi(hypothesis)
    ref_tokens = _tokenize_vi(reference)
    if not ref_tokens or not hyp_tokens:
        return {"precision": 0, "recall": 0, "f1": 0}

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0
    recall = lcs / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> float:
    hyp_tokens = _tokenize_vi(hypothesis)
    ref_tokens = _tokenize_vi(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        if not hyp_ng:
            precisions.append(0.0)
            continue
        clipped = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng)
        precisions.append(clipped / sum(hyp_ng.values()))

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    return bp * math.exp(log_avg)


def key_fact_coverage(answer: str, key_facts: list[str]) -> dict:
    if not key_facts:
        return {"coverage": 1.0, "found": [], "missed": []}

    answer_lower = answer.lower()
    found, missed = [], []
    for fact in key_facts:
        fact_words = _tokenize_vi(fact)
        if not fact_words:
            continue
        matched = sum(1 for word in fact_words if word in answer_lower)
        if matched / len(fact_words) >= 0.6:
            found.append(fact)
        else:
            missed.append(fact)
    coverage = len(found) / len(key_facts) if key_facts else 1.0
    return {"coverage": coverage, "found": found, "missed": missed}


def try_bertscore(hypotheses: list[str], references: list[str]) -> Optional[dict]:
    try:
        from bert_score import score as bert_score_fn

        p_vals, r_vals, f1_vals = bert_score_fn(
            hypotheses,
            references,
            lang="vi",
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        return {
            "precision": p_vals.mean().item(),
            "recall": r_vals.mean().item(),
            "f1": f1_vals.mean().item(),
            "per_sample_f1": [f.item() for f in f1_vals],
        }
    except ImportError:
        return None


def evaluate_generation(engine: RAGEngine, verbose: bool = True) -> dict:
    results_detail = []
    all_rouge1_f1, all_rouge2_f1, all_rouge_l_f1 = [], [], []
    all_bleu, all_coverage = [], []
    all_hypotheses, all_references = [], []

    if verbose:
        print("=" * 70)
        print("GENERATION QUALITY METRICS")
        print("=" * 70)

    for i, tc in enumerate(GENERATION_TESTS):
        query = tc["query"]
        golden = tc["golden_answer"]
        key_facts = tc.get("key_facts", [])

        if verbose:
            print(f"\n[{i + 1}/{len(GENERATION_TESTS)}] {query}")

        start_t = time.perf_counter()
        context = engine.retrieve(query, top_k=10)
        reranked = engine.rerank_results(query, context, top_k=5)
        reranked = engine.apply_metadata_boost(query, reranked)
        answer = engine.generate_answer(query, reranked)
        latency = (time.perf_counter() - start_t) * 1000

        r1 = rouge_n(answer, golden, n=1)
        r2 = rouge_n(answer, golden, n=2)
        rl = rouge_l(answer, golden)
        bl = bleu_score(answer, golden)
        kfc = key_fact_coverage(answer, key_facts)

        all_rouge1_f1.append(r1["f1"])
        all_rouge2_f1.append(r2["f1"])
        all_rouge_l_f1.append(rl["f1"])
        all_bleu.append(bl)
        all_coverage.append(kfc["coverage"])
        all_hypotheses.append(answer)
        all_references.append(golden)

        results_detail.append(
            {
                "query": query,
                "category": tc.get("category", ""),
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                "rouge1_f1": r1["f1"],
                "rouge2_f1": r2["f1"],
                "rougeL_f1": rl["f1"],
                "bleu": bl,
                "key_fact_coverage": kfc["coverage"],
                "missed_facts": kfc["missed"],
                "latency_ms": latency,
            }
        )

    bert_results = try_bertscore(all_hypotheses, all_references)
    summary = {
        "rouge1_f1": _avg(all_rouge1_f1),
        "rouge2_f1": _avg(all_rouge2_f1),
        "rougeL_f1": _avg(all_rouge_l_f1),
        "bleu": _avg(all_bleu),
        "key_fact_coverage": _avg(all_coverage),
        "bertscore_f1": bert_results["f1"] if bert_results else None,
    }

    categories = set(tc.get("category", "") for tc in GENERATION_TESTS)
    category_metrics = {}
    for cat in categories:
        cat_indices = [i for i, tc in enumerate(GENERATION_TESTS) if tc.get("category") == cat]
        if not cat_indices:
            continue
        category_metrics[cat] = {
            "count": len(cat_indices),
            "rouge1_f1": _avg([all_rouge1_f1[i] for i in cat_indices]),
            "rouge2_f1": _avg([all_rouge2_f1[i] for i in cat_indices]),
            "rougeL_f1": _avg([all_rouge_l_f1[i] for i in cat_indices]),
            "bleu": _avg([all_bleu[i] for i in cat_indices]),
            "key_fact_coverage": _avg([all_coverage[i] for i in cat_indices]),
        }

    return {
        "summary": summary,
        "category_metrics": category_metrics,
        "details": results_detail,
        "test_count": len(GENERATION_TESTS),
    }


# ============================================================
# Citation metrics
# ============================================================

VALID_ARTICLES = set(str(i) for i in range(1, 70))


def extract_cited_articles(text: str) -> list[str]:
    matches = re.findall(r"(?:[Dd]ieu|[Đđ]iều)\s+(\d+)", text)
    return list(set(matches))


def citation_precision(cited: list[str], must_cite: list[str]) -> float:
    if not cited:
        return 1.0 if not must_cite else 0.0
    correct = sum(1 for c in cited if c in must_cite)
    return correct / len(cited)


def citation_recall(cited: list[str], must_cite: list[str]) -> float:
    if not must_cite:
        return 1.0
    found = sum(1 for m in must_cite if m in cited)
    return found / len(must_cite)


def hallucination_check(cited: list[str], must_not_cite: list[str]) -> dict:
    fabricated = [c for c in cited if c not in VALID_ARTICLES]
    wrong_cite = [c for c in cited if c in must_not_cite]
    return {
        "fabricated_articles": fabricated,
        "wrong_citations": wrong_cite,
        "is_clean": len(fabricated) == 0 and len(wrong_cite) == 0,
    }


def evaluate_citation(engine: RAGEngine, verbose: bool = True) -> dict:
    results_detail = []
    all_precision, all_recall, all_clean, all_fabricated_count = [], [], [], []

    if verbose:
        print("=" * 70)
        print("CITATION AND ANTI-HALLUCINATION METRICS")
        print("=" * 70)

    for i, tc in enumerate(CITATION_TESTS):
        query = tc["query"]
        must_cite = tc.get("must_cite_articles", [])
        must_not_cite = tc.get("must_not_cite_articles", [])

        if verbose:
            print(f"\n[{i + 1}/{len(CITATION_TESTS)}] {query}")

        start_t = time.perf_counter()
        context = engine.retrieve(query, top_k=10)
        reranked = engine.rerank_results(query, context, top_k=5)
        reranked = engine.apply_metadata_boost(query, reranked)
        answer = engine.generate_answer(query, reranked)
        latency = (time.perf_counter() - start_t) * 1000

        cited = extract_cited_articles(answer)
        prec = citation_precision(cited, must_cite)
        rec = citation_recall(cited, must_cite)
        hall = hallucination_check(cited, must_not_cite)

        all_precision.append(prec)
        all_recall.append(rec)
        all_clean.append(1.0 if hall["is_clean"] else 0.0)
        all_fabricated_count.append(len(hall["fabricated_articles"]))

        results_detail.append(
            {
                "query": query,
                "category": tc.get("category", ""),
                "cited_articles": cited,
                "must_cite": must_cite,
                "must_not_cite": must_not_cite,
                "citation_precision": prec,
                "citation_recall": rec,
                "is_clean": hall["is_clean"],
                "fabricated": hall["fabricated_articles"],
                "wrong_citations": hall["wrong_citations"],
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                "latency_ms": latency,
            }
        )

    total = len(CITATION_TESTS)
    summary = {
        "citation_precision": _avg(all_precision),
        "citation_recall": _avg(all_recall),
        "clean_rate": _avg(all_clean),
        "hallucination_rate": 1.0 - _avg(all_clean),
        "total_fabricated": sum(all_fabricated_count),
        "avg_fabricated_per_query": _avg(all_fabricated_count),
    }

    categories = set(tc.get("category", "") for tc in CITATION_TESTS)
    category_metrics = {}
    for cat in categories:
        cat_indices = [i for i, tc in enumerate(CITATION_TESTS) if tc.get("category") == cat]
        if not cat_indices:
            continue
        category_metrics[cat] = {
            "count": len(cat_indices),
            "citation_precision": _avg([all_precision[i] for i in cat_indices]),
            "citation_recall": _avg([all_recall[i] for i in cat_indices]),
            "clean_rate": _avg([all_clean[i] for i in cat_indices]),
        }

    return {
        "summary": summary,
        "category_metrics": category_metrics,
        "details": results_detail,
        "test_count": total,
    }
