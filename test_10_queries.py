"""Run 10 SNP-disease queries through the MedEvi-Agent pipeline and report results."""
from __future__ import annotations

import json
import time
import sys
from typing import Any

from graph import app as graph_app


TEST_QUERIES = [
    {
        "query": "TP53 rs1042522 polymorphism and gastric cancer risk",
        "gene": "TP53",
        "rs": "rs1042522",
        "disease": "gastric cancer",
        "population": "gastric cancer patients",
        "intervention": "TP53 rs1042522 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "gastric cancer risk",
        "study_type": "case-control study",
    },
    {
        "query": "IL6 rs1800795 polymorphism and rheumatoid arthritis susceptibility",
        "gene": "IL6",
        "rs": "rs1800795",
        "disease": "rheumatoid arthritis",
        "population": "rheumatoid arthritis patients",
        "intervention": "IL6 rs1800795 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "rheumatoid arthritis susceptibility",
        "study_type": "case-control study",
    },
    {
        "query": "MTHFR rs1801133 polymorphism and ischemic stroke risk",
        "gene": "MTHFR",
        "rs": "rs1801133",
        "disease": "ischemic stroke",
        "population": "ischemic stroke patients",
        "intervention": "MTHFR rs1801133 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "ischemic stroke risk",
        "study_type": "case-control study",
    },
    {
        "query": "VEGFA rs2010963 polymorphism and diabetic retinopathy risk",
        "gene": "VEGFA",
        "rs": "rs2010963",
        "disease": "diabetic retinopathy",
        "population": "diabetic retinopathy patients",
        "intervention": "VEGFA rs2010963 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "diabetic retinopathy risk",
        "study_type": "case-control study",
    },
    {
        "query": "TNF rs1800629 polymorphism and inflammatory bowel disease susceptibility",
        "gene": "TNF",
        "rs": "rs1800629",
        "disease": "inflammatory bowel disease",
        "population": "inflammatory bowel disease patients",
        "intervention": "TNF rs1800629 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "inflammatory bowel disease susceptibility",
        "study_type": "case-control study",
    },
    {
        "query": "COX2 rs689466 polymorphism and colorectal cancer risk",
        "gene": "COX2",
        "rs": "rs689466",
        "disease": "colorectal cancer",
        "population": "colorectal cancer patients",
        "intervention": "COX2 rs689466 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "colorectal cancer risk",
        "study_type": "case-control study",
    },
    {
        "query": "ACE rs1799752 polymorphism and essential hypertension risk",
        "gene": "ACE",
        "rs": "rs1799752",
        "disease": "essential hypertension",
        "population": "essential hypertension patients",
        "intervention": "ACE rs1799752 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "essential hypertension risk",
        "study_type": "case-control study",
    },
    {
        "query": "SOD2 rs4880 polymorphism and coronary heart disease risk",
        "gene": "SOD2",
        "rs": "rs4880",
        "disease": "coronary heart disease",
        "population": "coronary heart disease patients",
        "intervention": "SOD2 rs4880 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "coronary heart disease risk",
        "study_type": "case-control study",
    },
    {
        "query": "CYP1A1 rs1048943 polymorphism and esophageal cancer risk",
        "gene": "CYP1A1",
        "rs": "rs1048943",
        "disease": "esophageal cancer",
        "population": "esophageal cancer patients",
        "intervention": "CYP1A1 rs1048943 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "esophageal cancer risk",
        "study_type": "case-control study",
    },
    {
        "query": "ERCC1 rs11615 polymorphism and non-small cell lung cancer risk",
        "gene": "ERCC1",
        "rs": "rs11615",
        "disease": "non-small cell lung cancer",
        "population": "non-small cell lung cancer patients",
        "intervention": "ERCC1 rs11615 variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": "non-small cell lung cancer risk",
        "study_type": "case-control study",
    },
]


def run_single(query_config: dict) -> dict:
    """Run a single query through the pipeline and return metrics."""
    init_state = {
        "query": query_config["query"],
        "pico_query": {
            "population": query_config["population"],
            "intervention": query_config["intervention"],
            "comparison": query_config["comparison"],
            "outcome": query_config["outcome"],
            "study_type": query_config["study_type"],
        },
        "pubmed_ids": [],
        "raw_abstracts": [],
        "reranked_abstracts": [],
        "screened_papers": [],
        "extracted_picos": [],
        "quantitative_outcomes": [],
        "grade_report": "",
        "error": "",
    }

    start = time.time()
    try:
        result = graph_app.invoke(init_state)
        elapsed = time.time() - start
    except Exception as exc:
        elapsed = time.time() - start
        return {
            "query": query_config["query"],
            "gene": query_config["gene"],
            "rs": query_config["rs"],
            "disease": query_config["disease"],
            "elapsed": round(elapsed, 1),
            "pubmed_count": 0,
            "screened_count": 0,
            "picos_count": 0,
            "quant_count": 0,
            "report_chars": 0,
            "error": str(exc),
            "status": "EXCEPTION",
        }

    # Extract metrics
    pubmed_count = len(result.get("pubmed_ids", []) or [])
    screened_count = len(result.get("screened_papers", []) or [])
    picos_count = len(result.get("extracted_picos", []) or [])
    quant_count = len(result.get("quantitative_outcomes", []) or [])
    report = result.get("grade_report", "") or ""
    error = result.get("error", "") or ""

    return {
        "query": query_config["query"],
        "gene": query_config["gene"],
        "rs": query_config["rs"],
        "disease": query_config["disease"],
        "elapsed": round(elapsed, 1),
        "pubmed_count": pubmed_count,
        "screened_count": screened_count,
        "picos_count": picos_count,
        "quant_count": quant_count,
        "report_chars": len(report),
        "error": error,
        "status": "OK" if screened_count > 0 and not error else ("WARN" if pubmed_count > 0 else "NO_RESULTS"),
    }


def main():
    print("=" * 90)
    print("MedEvi-Agent 10-SNP 测试集")
    print("=" * 90)
    print()

    results = []
    total_start = time.time()

    for i, qc in enumerate(TEST_QUERIES, 1):
        print(f"[{i:2d}/10] {qc['gene']} {qc['rs']} × {qc['disease']}")
        print(f"       Query: {qc['query']}")
        sys.stdout.flush()

        r = run_single(qc)
        results.append(r)

        icon = {"OK": "✅", "WARN": "⚠️", "NO_RESULTS": "❌", "EXCEPTION": "💥"}.get(r["status"], "?")
        print(f"       {icon} PubMed={r['pubmed_count']} | Screened={r['screened_count']} | "
              f"PICOS={r['picos_count']} | Quant={r['quant_count']} | "
              f"Report={r['report_chars']}chars | {r['elapsed']}s")
        if r["error"]:
            print(f"       Error: {r['error'][:120]}")
        print()
        sys.stdout.flush()

    total_elapsed = time.time() - total_start

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()

    # Per-query table
    header = f"{'#':<3} {'Gene':<8} {'RS':<12} {'Disease':<28} {'PubMed':<8} {'Screened':<10} {'PICOS':<8} {'Quant':<8} {'Report':<8} {'Time':<8} {'Status'}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        icon = {"OK": "✅", "WARN": "⚠️", "NO_RESULTS": "❌", "EXCEPTION": "💥"}.get(r["status"], "?")
        print(f"{i:<3} {r['gene']:<8} {r['rs']:<12} {r['disease']:<28} "
              f"{r['pubmed_count']:<8} {r['screened_count']:<10} {r['picos_count']:<8} "
              f"{r['quant_count']:<8} {r['report_chars']:<8} {r['elapsed']:<8.0f} {icon} {r['status']}")

    print()

    # Aggregate stats
    ok_count = sum(1 for r in results if r["status"] == "OK")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    no_count = sum(1 for r in results if r["status"] == "NO_RESULTS")
    exc_count = sum(1 for r in results if r["status"] == "EXCEPTION")

    pubmed_vals = [r["pubmed_count"] for r in results]
    screened_vals = [r["screened_count"] for r in results]
    picos_vals = [r["picos_count"] for r in results]
    quant_vals = [r["quant_count"] for r in results]
    time_vals = [r["elapsed"] for r in results]

    def stats(vals):
        return {
            "min": min(vals),
            "max": max(vals),
            "avg": round(sum(vals) / len(vals), 1),
            "sum": sum(vals),
        }

    print("Aggregate Statistics:")
    print(f"  Status      OK={ok_count}  WARN={warn_count}  NO_RESULTS={no_count}  EXCEPTION={exc_count}")
    print(f"  PubMed      min={stats(pubmed_vals)['min']}  max={stats(pubmed_vals)['max']}  "
          f"avg={stats(pubmed_vals)['avg']}  total={stats(pubmed_vals)['sum']}")
    print(f"  Screened    min={stats(screened_vals)['min']}  max={stats(screened_vals)['max']}  "
          f"avg={stats(screened_vals)['avg']}  total={stats(screened_vals)['sum']}")
    print(f"  PICOS       min={stats(picos_vals)['min']}  max={stats(picos_vals)['max']}  "
          f"avg={stats(picos_vals)['avg']}  total={stats(picos_vals)['sum']}")
    print(f"  Quant       min={stats(quant_vals)['min']}  max={stats(quant_vals)['max']}  "
          f"avg={stats(quant_vals)['avg']}  total={stats(quant_vals)['sum']}")
    print(f"  Time(s)     min={stats(time_vals)['min']}  max={stats(time_vals)['max']}  "
          f"avg={stats(time_vals)['avg']}  total={stats(time_vals)['sum']:.0f}")

    # Save to JSON for later analysis
    with open("test_results.json", "w") as f:
        json.dump({
            "total_elapsed_s": round(total_elapsed, 1),
            "summary": {
                "ok": ok_count,
                "warn": warn_count,
                "no_results": no_count,
                "exception": exc_count,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print()
    print("Results saved to test_results.json")

    # Return non-zero exit code if any failures
    if no_count > 0 or exc_count > 0:
        print(f"\n⚠️  {no_count + exc_count} queries had issues (no results or exceptions)")
        sys.exit(1)
    else:
        print("\n✅ All 10 queries completed successfully!")


if __name__ == "__main__":
    main()
