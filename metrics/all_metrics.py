
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULT_FILE_MAP: dict[str, str] = {
    "retrieval": "truy_xuat_ket_qua.json",
    "generation": "sinh_ket_qua.json",
    "citation": "trich_dan_ket_qua.json",
    "performance": "hieu_nang_ket_qua.json",
    "intent": "y_dinh_ket_qua.json",
}
COMBINED_REPORT_FILE = "tong_hop_bao_cao.json"


def run_intent(verbose: bool = True) -> dict:
    from metrics.system_metrics import evaluate_intent

    return evaluate_intent(verbose=verbose)


def run_retrieval(engine, verbose: bool = True) -> dict:
    from metrics.quality_metrics import evaluate_retrieval

    return evaluate_retrieval(engine, k_values=[3, 5, 10], verbose=verbose)


def run_generation(engine, verbose: bool = True) -> dict:
    from metrics.quality_metrics import evaluate_generation

    return evaluate_generation(engine, verbose=verbose)


def run_citation(engine, verbose: bool = True) -> dict:
    from metrics.quality_metrics import evaluate_citation

    return evaluate_citation(engine, verbose=verbose)


def run_performance(engine, verbose: bool = True) -> dict:
    from metrics.system_metrics import evaluate_performance

    return evaluate_performance(engine, n_runs=1, verbose=verbose)


MODULES: dict[str, dict[str, object]] = {
    "intent": {
        "fn": run_intent,
        "needs_engine": False,
        "description": "Intent Classification",
    },
    "retrieval": {
        "fn": run_retrieval,
        "needs_engine": True,
        "description": "Retrieval Quality",
    },
    "generation": {
        "fn": run_generation,
        "needs_engine": True,
        "description": "Generation Quality",
    },
    "citation": {
        "fn": run_citation,
        "needs_engine": True,
        "description": "Citation and Anti-Hallucination",
    },
    "performance": {
        "fn": run_performance,
        "needs_engine": True,
        "description": "System Performance",
    },
}

GEMINI_REQUIRED_MODULES = {"generation", "citation", "performance"}


def print_final_report(all_results: dict) -> None:
    print("\n" + "=" * 70)
    print("HUSC RAG CHATBOT - OVERALL EVALUATION REPORT")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if "intent" in all_results:
        s = all_results["intent"].get("summary", {})
        print("\nINTENT CLASSIFICATION")
        print(f"  Accuracy: {s.get('accuracy', 0):.1%} | Macro F1: {s.get('macro_f1', 0):.3f}")

    if "retrieval" in all_results:
        s = all_results["retrieval"].get("summary", {})
        print("\nRETRIEVAL QUALITY")
        print(
            f"  MRR: {s.get('mrr', 0):.3f} | P@5: {s.get('precision@5', 0):.3f} | "
            f"R@5: {s.get('recall@5', 0):.3f} | HR@5: {s.get('hit_rate@5', 0):.3f}"
        )

    if "generation" in all_results:
        s = all_results["generation"].get("summary", {})
        print("\nGENERATION QUALITY")
        print(
            f"  ROUGE-1: {s.get('rouge1_f1', 0):.3f} | ROUGE-L: {s.get('rougeL_f1', 0):.3f} | "
            f"BLEU: {s.get('bleu', 0):.3f} | KeyFacts: {s.get('key_fact_coverage', 0):.1%}"
        )
        bert = s.get("bertscore_f1")
        if bert is not None:
            print(f"  BERTScore F1: {bert:.3f}")

    if "citation" in all_results:
        s = all_results["citation"].get("summary", {})
        print("\nCITATION AND HALLUCINATION")
        print(
            f"  Precision: {s.get('citation_precision', 0):.3f} | "
            f"Recall: {s.get('citation_recall', 0):.3f} | "
            f"Clean Rate: {s.get('clean_rate', 0):.1%} | "
            f"Hallucination: {s.get('hallucination_rate', 0):.1%}"
        )

    if "performance" in all_results:
        s = all_results["performance"].get("summary", {})
        e2e = s.get("e2e_latency", {})
        print("\nPERFORMANCE")
        print(
            f"  E2E mean: {e2e.get('mean', 0):.0f}ms | "
            f"P90: {e2e.get('p90', 0):.0f}ms | "
            f"Throughput: {s.get('throughput_qps', 0):.2f} qps"
        )
        breakdown = s.get("latency_breakdown_pct", {})
        if breakdown:
            print(
                f"  Breakdown: Retrieve {breakdown.get('retrieve', 0):.0f}% | "
                f"Rerank {breakdown.get('rerank', 0):.0f}% | "
                f"Generate {breakdown.get('generate', 0):.0f}%"
            )

    print("=" * 70)


def run_metrics(
    only: list[str] | None = None,
    no_charts: bool = False,
    quiet: bool = False,
) -> dict:
    verbose = not quiet
    modules_to_run = only or list(MODULES.keys())

    needs_engine = any(bool(MODULES[m]["needs_engine"]) for m in modules_to_run)
    engine = None

    if needs_engine:
        print("Initializing RAGEngine...")
        from rag_core import RAGEngine

        engine = RAGEngine()
        engine.initialize(load_db=True)

        if engine.index is None or not engine.records:
            print("Database is not ready. Rebuild database before running engine-based metrics.")
            modules_to_run = [m for m in modules_to_run if not bool(MODULES[m]["needs_engine"])]
            if not modules_to_run:
                raise RuntimeError("No runnable modules left without a ready database.")
            print(f"Fallback to modules without engine: {modules_to_run}\n")
        else:
            print(f"Database loaded: {len(engine.records)} records\n")

    if (
        needs_engine
        and engine is not None
        and engine.index is not None
        and bool(engine.records)
        and any(m in GEMINI_REQUIRED_MODULES for m in modules_to_run)
    ):
        requested_gemini = [m for m in modules_to_run if m in GEMINI_REQUIRED_MODULES]
        print("Checking Gemini API availability...")
        gemini_ok = False
        try:
            gemini_ok = bool(engine.configure_gemini())
        except Exception as exc:
            print(f"Gemini check failed: {exc}")

        if gemini_ok:
            print("Gemini API is available.\n")
        else:
            modules_to_run = [m for m in modules_to_run if m not in GEMINI_REQUIRED_MODULES]
            print(f"Skip Gemini-required modules: {', '.join(requested_gemini)}")
            if not modules_to_run:
                raise RuntimeError("No runnable modules left because Gemini API is unavailable.")
            print(f"Continue with: {modules_to_run}\n")

    all_results: dict = {}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total_start = time.perf_counter()

    for module_name in modules_to_run:
        module_def = MODULES[module_name]
        module_fn = module_def["fn"]
        module_desc = module_def["description"]
        module_needs_engine = bool(module_def["needs_engine"])

        print("\n" + "-" * 70)
        print(f"Running: {module_desc} ({module_name})")
        print("-" * 70)

        start_t = time.perf_counter()
        try:
            if module_needs_engine:
                result = module_fn(engine, verbose=verbose)  # type: ignore[misc]
            else:
                result = module_fn(verbose=verbose)  # type: ignore[misc]

            elapsed = time.perf_counter() - start_t
            result["_elapsed_seconds"] = round(elapsed, 2)
            all_results[module_name] = result.get("summary", result)

            result_filename = RESULT_FILE_MAP.get(module_name, f"{module_name}_results.json")
            path = os.path.join(RESULTS_DIR, result_filename)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2, default=str)
            print(f"{module_name} done ({elapsed:.1f}s) -> {path}")
        except Exception as exc:
            elapsed = time.perf_counter() - start_t
            print(f"{module_name} failed ({elapsed:.1f}s): {exc}")
            all_results[module_name] = {"error": str(exc)}

    total_elapsed = time.perf_counter() - total_start
    report = {
        "timestamp": datetime.now().isoformat(),
        "modules_run": modules_to_run,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "results": all_results,
    }
    report_path = os.path.join(RESULTS_DIR, COMBINED_REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=str)

    print_final_report(all_results)
    print(f"\nCombined report: {report_path}")
    print(f"Total runtime: {total_elapsed:.1f}s")

    if not no_charts:
        print("\nGenerating charts...")
        try:
            from metrics.visualize_metrics import generate_all_charts

            generate_all_charts(verbose=verbose)
        except Exception as exc:
            print(f"Chart generation failed: {exc}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all HUSC RAG evaluation metrics")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(MODULES.keys()),
        help="Run only selected modules",
    )
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--quiet", action="store_true", help="Reduce log output")
    args = parser.parse_args()

    run_metrics(only=args.only, no_charts=args.no_charts, quiet=args.quiet)


if __name__ == "__main__":
    main()
