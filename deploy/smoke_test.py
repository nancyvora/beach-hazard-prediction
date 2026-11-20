"""Check a locally running API or an explicitly supplied deployed URL."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from src.common import read_json, write_json


def check(url, beach, date, benchmark=False):
    with httpx.Client(base_url=url, timeout=30) as client:
        health = client.get("/health")
        health.raise_for_status()
        prediction = client.get("/predict", params={"beach": beach, "date": date})
        prediction.raise_for_status()
        body = prediction.json()
        assert body["stations"] and 0 <= body["max_station_probability"] <= 1
        assert body["disclaimer"]
        for key in ["model_version", "feature_version", "history_snapshot_id"]:
            assert body[key] == health.json()[key]
        assert client.get("/predict", params={"beach": 99999999, "date": date}).status_code == 404
        assert client.get("/predict", params={"beach": beach, "date": "bad"}).status_code == 422
        result = {"status": "passed", "model_version": body["model_version"], "demo": body["demo"]}
        if benchmark:

            def timed_request(_):
                started = time.perf_counter()
                response = client.get("/predict", params={"beach": beach, "date": date})
                response.raise_for_status()
                return time.perf_counter() - started

            sequential = [timed_request(i) for i in range(20)]
            with ThreadPoolExecutor(max_workers=10) as pool:
                parallel = list(pool.map(timed_request, range(50)))
            result.update(
                sequential_p95_seconds=float(np.percentile(sequential, 95)),
                parallel_p95_seconds=float(np.percentile(parallel, 95)),
                sequential_requests=20,
                parallel_requests=50,
                failures=0,
            )
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--beach", type=int)
    parser.add_argument("--date")
    parser.add_argument("--case", type=Path, default=Path("reports/deploy/smoke_case.json"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.beach is None or args.date is None:
        case = read_json(args.case)
        args.beach, args.date = case["beach"], case["date"]
    result = check(args.url, args.beach, args.date, args.benchmark)
    if args.output:
        write_json(args.output, result)
    print(result)
