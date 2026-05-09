import json
from pathlib import Path

path = Path("data/shl_product_catalog_clean.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total items:", len(data))
print("\nFirst item:")
print(json.dumps(data[0], indent=2, ensure_ascii=False))
