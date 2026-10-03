import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from auditor.collector import fetch_all_engines

load_dotenv()

query = "Is the sun a planet?"
results = fetch_all_engines(query)

for r in results:
    print("=" * 50)
    print("Engine:", r.engine)
    print("Error:", r.error)
    print("Header:", r.header)
    print("Text blocks:", len(r.text_blocks))
    print("References:", len(r.references))