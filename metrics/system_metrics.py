
from __future__ import annotations

import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent_classifier import classify_intent
from rag_core import RAGEngine
from metrics.evaluation_dataset import INTENT_TESTS, PERFORMANCE_QUERIES

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_intent(verbose: bool = True) -> dict:
    results_detail = []
    correct = 0
    total = len(INTENT_TESTS)

    classes = ["small_talk", "out_of_scope", "academic"]
    counters = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in classes}
    confusion = {actual: {pred: 0 for pred in classes} for actual in classes}
    confidence_by_class = defaultdict(list)
    misclassified = []

    if verbose:
        print("=" * 70)
        print("INTENT CLASSIFICATION METRICS")
        print("=" * 70)

    for case in INTENT_TESTS:
        query = case["query"]
        expected = case["expected_intent"]
        result = classify_intent(query)
        predicted = result.intent.value

        is_correct = predicted == expected
        if is_correct:
            correct += 1
            counters[expected]["tp"] += 1
        else:
            counters[expected]["fn"] += 1
            counters[predicted]["fp"] += 1
            misclassified.append(
                {
                    "query": query,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": result.confidence,
                    "matched_pattern": result.matched_pattern,
                }
            )

        confusion[expected][predicted] += 1
        confidence_by_class[expected].append(result.confidence)
        results_detail.append(
            {
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "confidence": result.confidence,
                "matched_pattern": result.matched_pattern,
            }
        )

    per_class = {}
    for cls in classes:
        tp = counters[cls]["tp"]
        fp = counters[cls]["fp"]
        fn = counters[cls]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "avg_confidence": _avg(confidence_by_class[cls]),
        }

    summary = {
        "accuracy": correct / total if total > 0 else 0,
        "macro_f1": _avg([per_class[cls]["f1"] for cls in classes]),
        "total": total,
        "correct": correct,
        "misclassified_count": len(misclassified),
    }

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "misclassified": misclassified,
        "details": results_detail,
        "test_count": total,
    }


def get_memory_mb() -> Optional[float]:
    if not HAS_PSUTIL:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def evaluate_performance(engine: RAGEngine, n_runs: int = 1, verbose: bool = True) -> dict:
    queries = PERFORMANCE_QUERIES
    results_detail = []

    all_retrieve, all_rerank, all_generate, all_e2e = [], [], [], []
    mem_before = get_memory_mb()

    if verbose:
        print("=" * 70)
        print("SYSTEM PERFORMANCE METRICS")
        print("=" * 70)

    for query in queries:
        run_retrieve, run_rerank, run_generate, run_e2e = [], [], [], []
        answer = ""
        for _ in range(n_runs):
            t0 = time.perf_counter()
            context = engine.retrieve(query, top_k=20)
            t1 = time.perf_counter()
            reranked = engine.rerank_results(query, context, top_k=5)
            reranked = engine.apply_metadata_boost(query, reranked)
            t2 = time.perf_counter()
            answer = engine.generate_answer(query, reranked)
            t3 = time.perf_counter()

            run_retrieve.append((t1 - t0) * 1000)
            run_rerank.append((t2 - t1) * 1000)
            run_generate.append((t3 - t2) * 1000)
            run_e2e.append((t3 - t0) * 1000)

        avg_retrieve = statistics.mean(run_retrieve)
        avg_rerank = statistics.mean(run_rerank)
        avg_generate = statistics.mean(run_generate)
        avg_e2e = statistics.mean(run_e2e)

        all_retrieve.append(avg_retrieve)
        all_rerank.append(avg_rerank)
        all_generate.append(avg_generate)
        all_e2e.append(avg_e2e)

        results_detail.append(
            {
                "query": query,
                "retrieve_ms": round(avg_retrieve, 1),
                "rerank_ms": round(avg_rerank, 1),
                "generate_ms": round(avg_generate, 1),
                "e2e_ms": round(avg_e2e, 1),
                "answer_length": len(answer),
            }
        )

    mem_after = get_memory_mb()

    def _stats(values: list[float]) -> dict:
        if not values:
            return {}
        return {
            "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "p90": round(sorted(values)[int(len(values) * 0.9)], 1) if len(values) >= 2 else round(values[0], 1),
            "p95": round(sorted(values)[int(len(values) * 0.95)], 1) if len(values) >= 2 else round(values[0], 1),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
            "std": round(statistics.stdev(values), 1) if len(values) >= 2 else 0,
        }

    total_time_s = sum(all_e2e) / 1000
    throughput = len(queries) / total_time_s if total_time_s > 0 else 0

    summary = {
        "e2e_latency": _stats(all_e2e),
        "retrieve_latency": _stats(all_retrieve),
        "rerank_latency": _stats(all_rerank),
        "generate_latency": _stats(all_generate),
        "throughput_qps": round(throughput, 3),
        "total_queries": len(queries),
        "total_time_s": round(total_time_s, 2),
        "memory_before_mb": round(mem_before, 1) if mem_before else None,
        "memory_after_mb": round(mem_after, 1) if mem_after else None,
        "memory_delta_mb": round(mem_after - mem_before, 1) if (mem_before and mem_after) else None,
    }

    mean_e2e = summary["e2e_latency"]["mean"]
    if mean_e2e > 0:
        summary["latency_breakdown_pct"] = {
            "retrieve": round(summary["retrieve_latency"]["mean"] / mean_e2e * 100, 1),
            "rerank": round(summary["rerank_latency"]["mean"] / mean_e2e * 100, 1),
            "generate": round(summary["generate_latency"]["mean"] / mean_e2e * 100, 1),
        }

    return {
        "summary": summary,
        "details": results_detail,
        "test_count": len(queries),
    }

