"""Command-line interface for the AI Answer Auditor."""

import argparse
import os
import sys

import serpapi
from dotenv import load_dotenv

from auditor.collector import fetch_all_engines
from auditor.extractor import extract_claims
from auditor.verifier import verify_claims
from auditor.arbiter import arbitrate

MAX_CLAIMS_PER_ENGINE = 6


def run_audit(query: str) -> None:
    load_dotenv()

    print(f"\nQuestion: {query}")
    print("Collecting answers from AI search engines...")

    engine_answers = fetch_all_engines(query)
    serpapi_client = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"], timeout=60)

    engine_results = {}

    for answer in engine_answers:
        if answer.error:
            print(f"  [{answer.engine}] failed: {answer.error}")
            continue

        extracted = extract_claims(answer.engine, answer.text_blocks)
        verified = verify_claims(
            extracted.claims, serpapi_client, max_claims=MAX_CLAIMS_PER_ENGINE
        )
        engine_results[answer.engine] = verified
        print(f"  [{answer.engine}] {len(verified)} claims checked")

    print("Synthesizing consensus answer...\n")
    final = arbitrate(query, engine_results)

    print("=" * 60)
    print("AUDITED ANSWER")
    print("=" * 60)
    print(final.consensus_answer)
    print(f"\nConfidence: {final.confidence.upper()}")

    if final.agreements:
        print("\nWhat the evidence supports:")
        for a in final.agreements:
            print(f"  + {a}")

    if final.disagreements:
        print("\nDisagreements found:")
        for d in final.disagreements:
            print(f"  ! {d}")

    if final.abstained:
        print("\nNote: confidence was too low for a definitive answer.")

    print()


def main():
    parser = argparse.ArgumentParser(description="AI Answer Auditor")
    parser.add_argument("query", nargs="*", help="The question to audit")
    args = parser.parse_args()

    if args.query:
        query = " ".join(args.query)
    else:
        query = input("Enter a question to audit: ").strip()

    if not query:
        print("No question provided.")
        sys.exit(1)

    run_audit(query)


if __name__ == "__main__":
    main()