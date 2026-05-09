import json
from pathlib import Path

input_path = Path("data/shl_product_catalog.json")
output_path = Path("data/shl_product_catalog_clean.json")

raw = input_path.read_text(encoding="utf-8", errors="replace")

print("Original file size:", len(raw), "characters")

try:
    data = json.loads(raw)
    print("Loaded normally.")
except json.JSONDecodeError as e:
    print("Normal JSON load failed:")
    print(e)
    print("Trying strict=False...")
    data = json.loads(raw, strict=False)
    print("Loaded using strict=False.")

output_path.write_text(
    json.dumps(data, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Clean JSON saved to:", output_path)
print("Top-level type:", type(data))

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
elif isinstance(data, list):
    print("List length:", len(data))
