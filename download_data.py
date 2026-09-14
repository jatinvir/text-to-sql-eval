"""Fetch Spider dev (validation.json) and only the 20 databases it uses.

premai-io/spider rather than xlangai/spider: the latter has no SQLite files, and execution
is the whole grading strategy. The full repo is ~919 MB, mostly a train-only database
(soccer_1), so databases are selected by the db_ids that appear in validation.json.

    python download_data.py
"""

import json

from huggingface_hub import hf_hub_download, snapshot_download

from harness.paths import DATA_DIR, VALIDATION_JSON

REPO_ID = "premai-io/spider"
# Pinned so a fresh download is byte-identical to what cases.yaml was built from.
REVISION = "8698f764f0547eb7136b3be9569093c7f4217be4"


def main():
    hf_hub_download(REPO_ID, "validation.json", repo_type="dataset", revision=REVISION, local_dir=DATA_DIR)
    with open(VALIDATION_JSON) as f:
        db_ids = sorted({rec["db_id"] for rec in json.load(f)})
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=DATA_DIR,
        allow_patterns=[f"database/{db_id}/*" for db_id in db_ids],
    )
    missing = [d for d in db_ids if not (DATA_DIR / "database" / d / f"{d}.sqlite").exists()]
    if missing:
        raise SystemExit(f"download incomplete, missing: {missing}")
    print(f"validation.json + {len(db_ids)} databases in {DATA_DIR}")


if __name__ == "__main__":
    main()
