import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from auditor.extractor import extract_claims

load_dotenv()

path = pathlib.Path("fixtures/raw/bing_copilot__what_is_the_current_repo_rate_set_by_the_reserve_b.json")
data = json.loads(path.read_text(encoding="utf-8"))

result = extract_claims("bing_copilot", data["text_blocks"])

print("Error:", result.error)
print("Claims found:", len(result.claims))
for c in result.claims:
    print(" -", c)