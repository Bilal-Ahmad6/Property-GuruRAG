"""Benchmark Groq chat performance and basic relevance.

Uses answer_one_web from scripts.query_rag to issue sample queries and
measures latency plus simple heuristic relevance checks.

Outputs:
 - Console table summary
 - JSON lines file at logs/benchmark_results.jsonl
"""
from __future__ import annotations

import os
import time
import json
import sys
from dataclasses import dataclass, asdict
import argparse
from pathlib import Path
from typing import List, Optional

# Ensure the scripts directory is on sys.path so we can import query_rag when executed from project root
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from query_rag import answer_one_web, rag_infer  # type: ignore

LOG_PATH = Path("logs/benchmark_results.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

SAMPLE_QUERIES = [
    # (query, expected_property_count or None)
    ("Hi", None),
    ("Show me 2 houses under 5 crore", 2),
    ("Find 1 apartment with 2 bedrooms", 1),
    ("Average price of 10 marla houses", None),
    ("Tell me a joke about cats", 0),  # Irrelevant query should not list properties
]

PROPERTY_KEYWORDS = {"house", "houses", "apartment", "apartments", "property", "properties", "listing", "listings"}

@dataclass
class Result:
    query: str
    latency_ms: float
    answer: str
    word_count: int
    detected_property_blocks: int
    expected_property_count: Optional[int]
    property_count_match: Optional[bool]
    contains_price_token: bool
    relevance_pass: bool
    notes: str


def detect_property_blocks(answer: str) -> int:
    # Count occurrences of pattern "Property #"
    return answer.count("Property #")


def contains_price(answer: str) -> bool:
    return any(tok in answer for tok in ["PKR", "price", "Price:"])


from typing import Tuple


def evaluate_relevance(query: str, answer: str, expected_count: Optional[int]) -> Tuple[bool, str, int, Optional[bool]]:
    prop_blocks = detect_property_blocks(answer)
    price_flag = contains_price(answer)

    # Greeting / casual
    if query.lower() in {"hi", "hello", "hey"}:
        if prop_blocks == 0:
            return True, "Greeting handled (no listings)", prop_blocks, None
        return False, "Unexpected listings in greeting", prop_blocks, None

    # Irrelevant query: expect zero listings
    if "joke" in query.lower() or "cat" in query.lower():
        if prop_blocks == 0:
            return True, "Irrelevant filtered", prop_blocks, None
        return False, "Irrelevant query returned listings", prop_blocks, None

    # Statistical query: expect some numeric insight, listings optional
    if any(k in query.lower() for k in ["average", "mean", "avg"]):
        if any(ch.isdigit() for ch in answer):  # crude numeric presence
            return True, "Stats answer has numbers", prop_blocks, None
        return False, "Stats answer missing numbers", prop_blocks, None

    # Property search with expected count
    if expected_count is not None and expected_count > 0:
        match = (prop_blocks == expected_count)
        if match and price_flag:
            return True, "Property count and price present", prop_blocks, match
        if match and not price_flag:
            return True, "Property count ok (price token absent)", prop_blocks, match
        return False, f"Expected {expected_count} property blocks, got {prop_blocks}", prop_blocks, match

    # Fallback heuristic: if property keywords in query, expect at least one block
    if any(w in query.lower() for w in PROPERTY_KEYWORDS):
        if prop_blocks > 0:
            return True, "Listings returned", prop_blocks, None
        return False, "No listings found for property query", prop_blocks, None

    return True, "No specific relevance rule applied", prop_blocks, None


def benchmark(groq_api_key: str, groq_model: str = "llama3-8b-8192") -> List[Result]:
    results: List[Result] = []
    for query, expected_count in SAMPLE_QUERIES:
        start = time.perf_counter()
        # Use structured RAG inference so we could in the future inspect retrieved distances, etc.
        structured = rag_infer(query=query, groq_api_key=groq_api_key, groq_model=groq_model)
        answer = str(structured.get("answer", ""))
        latency = (time.perf_counter() - start) * 1000.0
        word_count = len(answer.split()) if answer else 0
        relevance_pass, notes, prop_blocks, count_match = evaluate_relevance(query, answer, expected_count)
        res = Result(
            query=query,
            latency_ms=latency,
            answer=answer[:500].replace("\n", " ") + ("..." if len(answer) > 500 else ""),
            word_count=word_count,
            detected_property_blocks=prop_blocks,
            expected_property_count=expected_count,
            property_count_match=count_match,
            contains_price_token=contains_price(answer),
            relevance_pass=relevance_pass,
            notes=notes,
        )
        results.append(res)
    return results


def print_table(results: List[Result]) -> None:
    headers = ["Query", "Latency ms", "Words", "Props", "Exp", "Match", "Price?", "Rel?", "Notes"]
    rows = []
    for r in results:
        rows.append([
            r.query,
            f"{r.latency_ms:.0f}",
            str(r.word_count),
            str(r.detected_property_blocks),
            "-" if r.expected_property_count is None else str(r.expected_property_count),
            "-" if r.property_count_match is None else ("Y" if r.property_count_match else "N"),
            "Y" if r.contains_price_token else "N",
            "PASS" if r.relevance_pass else "FAIL",
            r.notes,
        ])

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    def fmt_row(cols):
        return " | ".join(str(c).ljust(w) for c, w in zip(cols, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def save_jsonl(results: List[Result]) -> None:
    with LOG_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"\nSaved detailed results to {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Groq-backed RAG chat")
    parser.add_argument("--groq-api-key", dest="groq_api_key", help="Groq API key (overrides env)")
    parser.add_argument("--groq-model", dest="groq_model", default="llama3-8b-8192", help="Groq model name")
    args = parser.parse_args()

    groq_api_key = (
        args.groq_api_key
        or os.getenv("GROQ_API_KEY")
        or os.getenv("ZAMEEN_GROQ_API_KEY")
    )
    if not groq_api_key:
        # Fallback to config settings if available
        try:
            from config import settings  # type: ignore
            groq_api_key = settings.groq_api_key or groq_api_key
        except Exception:
            pass

    if not groq_api_key:
        print("Error: GROQ_API_KEY not provided (use --groq-api-key or set env variable).")
        return 1

    print(f"Running Groq chat benchmark on sample queries (model={args.groq_model})...")
    results = benchmark(groq_api_key, groq_model=args.groq_model)
    print_table(results)

    # Optional aggregate latency stats (based on overall measured latency per query)
    if results:
        latencies = [r.latency_ms for r in results]
        print("\nLatency summary (ms): min={:.0f} p50={:.0f} avg={:.0f} p95={:.0f} max={:.0f}".format(
            min(latencies),
            sorted(latencies)[len(latencies)//2],
            sum(latencies)/len(latencies),
            sorted(latencies)[int(len(latencies)*0.95)-1],
            max(latencies)
        ))
    save_jsonl(results)

    passed = sum(1 for r in results if r.relevance_pass)
    print(f"\nSummary: {passed}/{len(results)} relevance checks passed ({passed/len(results)*100:.1f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
