import re
import os
import sys
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# 1. Hook this up to your real pipeline
# ---------------------------------------------------------------------------
def query_rag(question: str):
    """
    Calls into the actual RAG pipeline.
    """
    from src.rag_chain import ask_question
    result = ask_question(question)
    
    # Map the pipeline's 'filename' key to the 'source' key expected by the eval script
    # and use basename to ensure we match "b8.pdf" instead of "data/raw/b8.pdf"
    sources = [{"source": os.path.basename(s.get("filename", ""))} for s in result.get("sources", [])]
    return {"answer": result["answer"], "sources": sources}


# ---------------------------------------------------------------------------
# 2. Test case definitions
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    name: str
    question: str
    # substrings that MUST appear (case-insensitive) in the answer for it to pass
    must_contain: list[str] = field(default_factory=list)
    # substrings that must NOT appear (e.g. hallucinated facts)
    must_not_contain: list[str] = field(default_factory=list)
    # source filenames that retrieval should include (e.g. {"b8.pdf"})
    expected_sources: set[str] = field(default_factory=set)
    # if True, ALL expected_sources must appear; if False, ANY one is enough
    require_all_sources: bool = True
    # custom check function, receives the full result dict, returns (bool, reason)
    custom_check: Callable | None = None


TEST_SUITE: list[TestCase] = [
    TestCase(
        name="single_doc_factual_cardiology",
        question="What were the reductions in mortality and hospitalization seen with carvedilol in the US Carvedilol Heart Failure Group trial?",
        must_contain=["3.2", "7.8"],
        expected_sources={"b8.pdf"},
    ),
    TestCase(
        name="numeric_table_lookup_cancer",
        question="In the breast cancer radiotherapy study, what was the hazard ratio for ischaemic heart disease death at 15+ years in women who received left-side radiotherapy compared to non-irradiated right-sided patients?",
        must_contain=["1.59"],
        expected_sources={"b9.pdf"},
    ),
    TestCase(
        name="cross_document_comparison",
        question="Both documents discuss cardiovascular risk from medical interventions. Compare the cardiac risks discussed: one paper covers treatments to prevent cardiac events, the other covers a cancer treatment that increases cardiac risk. What are the key differences in what's being measured?",
        expected_sources={"b8.pdf", "b9.pdf"},
        require_all_sources=True,  # THE key regression test for query decomposition
    ),
    TestCase(
        name="negative_control_hallucination_check",
        question="What did these papers say about the impact of quantum computing on stock prices?",
        must_not_contain=["quantum"],  # answer should NOT claim the papers discuss quantum computing
        custom_check=lambda result: (
            any(
                phrase in result["answer"].lower()
                for phrase in ["cannot answer", "not covered", "no mention", "does not discuss", "no information", "provided documents do not mention", "not mentioned"]
            ),
            "Expected a decline/no-info response; bot may have hallucinated.",
        ),
    ),
    TestCase(
        name="single_doc_factual_cardiology_2",
        question="What treatments does the Cardiology paper recommend for preventing acute myocardial infarction?",
        must_contain=["abciximab"],
        expected_sources={"b8.pdf"},
    ),
]


# ---------------------------------------------------------------------------
# 3. Evaluation logic
# ---------------------------------------------------------------------------
def get_source_filenames(result: dict) -> set[str]:
    return {s.get("source", "").strip() for s in result.get("sources", [])}


def run_test(tc: TestCase) -> dict:
    result = query_rag(tc.question)
    answer_lower = result["answer"].lower()
    found_sources = get_source_filenames(result)

    failures = []

    for phrase in tc.must_contain:
        if phrase.lower() not in answer_lower:
            failures.append(f"Missing expected content: '{phrase}'")

    for phrase in tc.must_not_contain:
        if phrase.lower() in answer_lower:
            failures.append(f"Found forbidden content: '{phrase}'")

    if tc.expected_sources:
        if tc.require_all_sources:
            missing = tc.expected_sources - found_sources
            if missing:
                failures.append(f"Missing expected source(s): {missing} (got: {found_sources})")
        else:
            if not (tc.expected_sources & found_sources):
                failures.append(f"None of expected sources {tc.expected_sources} found (got: {found_sources})")

    if tc.custom_check:
        passed, reason = tc.custom_check(result)
        if not passed:
            failures.append(reason)

    return {
        "name": tc.name,
        "question": tc.question,
        "answer": result["answer"],
        "sources": sorted(found_sources),
        "passed": len(failures) == 0,
        "failures": failures,
    }


def main():
    print("=" * 70)
    print("RAG EVALUATION SUITE")
    print("=" * 70)

    results = []
    for tc in TEST_SUITE:
        import time
        time.sleep(15)  # Pace requests to avoid Gemini 15 RPM free tier rate limit
        try:
            r = run_test(tc)
        except NotImplementedError as e:
            print(f"\n[SETUP REQUIRED] {e}")
            sys.exit(1)
        except Exception as e:
            r = {
                "name": tc.name,
                "question": tc.question,
                "answer": "",
                "sources": [],
                "passed": False,
                "failures": [f"Exception during query: {e}"],
            }
        results.append(r)

    passed_count = sum(r["passed"] for r in results)

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n{status}  {r['name']}")
        print(f"   Q: {r['question'][:90]}{'...' if len(r['question']) > 90 else ''}")
        print(f"   Sources retrieved: {r['sources']}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"   ⚠️  {f}")

    print("\n" + "=" * 70)
    print(f"RESULT: {passed_count}/{len(results)} tests passed")
    print("=" * 70)

    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    main()
