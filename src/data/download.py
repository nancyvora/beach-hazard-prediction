"""Read-only access to the official resource. Downloads are explicit, never part of tests."""

import argparse
import csv
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from src.common import write_json

RESOURCE_ID = "7bd961cf-abe4-433b-8033-378161237ff3"
API = "https://publish.data.ca.gov/api/3/action/datastore_search"
CSV_URL = (
    "https://publish.data.ca.gov/dataset/b9c8ce91-40ff-4ad3-8164-bc17c46afb44/"
    f"resource/{RESOURCE_ID}/download/beach-monitoring-results.csv"
)


def download(output, limit=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve existing raw data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if limit is None:
        with urllib.request.urlopen(CSV_URL, timeout=180) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
    else:
        if not 1 <= limit <= 100000:
            raise ValueError("API sample limit must be 1..100000; use --full for the CSV")
        offset, total = 0, None
        with temporary.open("w", newline="") as target:
            writer = None
            while offset < limit:
                query = urllib.parse.urlencode(
                    {
                        "resource_id": RESOURCE_ID,
                        "limit": min(1000, limit - offset),
                        "offset": offset,
                        "sort": "_id asc",
                    }
                )
                with urllib.request.urlopen(API + "?" + query, timeout=90) as response:
                    payload = json.load(response)
                if not payload.get("success"):
                    raise RuntimeError("Official Data API returned failure")
                result = payload["result"]
                records = result["records"]
                total = result["total"]
                if writer is None:
                    fields = [f["id"] for f in result["fields"] if f["id"] != "_id"]
                    writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                if not records:
                    break
                writer.writerows(records)
                offset += len(records)
        write_json(
            output.with_suffix(".metadata.json"),
            {
                "resource_id": RESOURCE_ID,
                "rows_downloaded": offset,
                "resource_total": total,
                "scope": "first API rows, NOT random or representative",
            },
        )
    temporary.replace(output)
    print(f"Saved {output}; scope={'full CSV' if limit is None else 'API sample'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="download the full large CSV (~1.7 GB)")
    group.add_argument(
        "--limit", type=int, help="API preview only; not suitable for population estimates"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        "data/raw/beach-monitoring-results.csv" if args.full else "data/raw/api_sample.csv"
    )
    download(output, None if args.full else args.limit)
