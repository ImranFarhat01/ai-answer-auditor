import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import serpapi
from auditor.verifier import verify_claim

load_dotenv()

client = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"], timeout=60)

claim = "The Reserve Bank of India repo rate is 5.25% as of October 2026."
result = verify_claim(claim, client)

print("Claim:", result.claim)
print("Verdict:", result.verdict)
print("Reason:", result.reason)
print("Best source:", result.best_source)
print("Error:", result.error)