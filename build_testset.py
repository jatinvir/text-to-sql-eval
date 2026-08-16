import json
import os
from vendor.spider_eval.hardness import eval_hardness

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "data", "validation.json")

with open(path, "r") as f:
    validation_data = json.load(f)

counts = {}
for record in validation_data:
    level = eval_hardness(record["sql"])
    counts[level] = counts.get(level, 0) + 1

print(counts)
