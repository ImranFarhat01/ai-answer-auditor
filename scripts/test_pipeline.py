import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import serpapi
from auditor.collector import fetch_all_engines
from auditor.extractor import extract_claims
from auditor.verifier import verify_claims
from auditor.arbiter import arbitrate

load_dotenv()

QUERY = "What is the current repo rate set by the Reserve Bank of India?"

print("Step 1: collecting engine answers...")
engine_answers = fetch_all_engines(QUERY)

serpapi_client = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"], timeout=60)

engine_results = {}

for answer in engine_answers:
    print(f"\n--- {answer.engine} ---")
    if answer.error:
        print("  Error fetching answer:", answer.error)
        continue

    print("Step 2: extracting claims...")
    extracted = extract_claims(answer.engine, answer.text_blocks)
    print(f"  {len(extracted.claims)} claims extracted")
    for c in extracted.claims:
        print("   -", c)

    print("Step 3: verifying claims (max 3 for this test)...")
    verified = verify_claims(extracted.claims, serpapi_client, max_claims=5)
    for v in verified:
        print(f"   [{v.verdict}] {v.claim}")

    engine_results[answer.engine] = verified

print("\nStep 4: arbitrating final answer...")
final = arbitrate(QUERY, engine_results)

print("\n" + "=" * 50)
print("FINAL CONSENSUS ANSWER")
print("=" * 50)
print("Answer:", final.consensus_answer)
print("Confidence:", final.confidence)
print("Abstained:", final.abstained)
print("Agreements:", final.agreements)
print("Disagreements:", final.disagreements)
print("Error:", final.error)