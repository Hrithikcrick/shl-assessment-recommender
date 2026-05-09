import json
from pathlib import Path

path = Path("data/shl_product_catalog_clean.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level type:", type(data))

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    for k, v in data.items():
        print(k, type(v))
        if isinstance(v, list):
            print("List length:", len(v))
            if len(v) > 0:
                print("First item keys:", list(v[0].keys()) if isinstance(v[0], dict) else type(v[0]))
            break

elif isinstance(data, list):
    print("List length:", len(data))
    if len(data) > 0:
        print("First item type:", type(data[0]))
        if isinstance(data[0], dict):
            print("First item keys:", list(data[0].keys()))
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:2000])
