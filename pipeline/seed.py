"""Seed the raw/ directory with historical USGS earthquake CSVs."""

import argparse
import csv
import os
import time
import datetime as dt
import requests

BASE = "https://earthquake.usgs.gov/fdsnws/event/1"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
LIMIT = 20_000
MIN_WINDOW = dt.timedelta(hours=1)
CUTOFF_LAG = dt.timedelta(days=2)
REQUEST_TIMEOUT = 120
COURTESY_DELAY = 0.5
MAX_RETRIES = 5


def fmt(d):
    return d.strftime("%Y-%m-%dT%H:%M:%S")


def filename(start, end):
    tag = f"{start:%Y%m%d_%H%M}-{end:%Y%m%d_%H%M}.csv"
    return os.path.join(RAW_DIR, tag)


def fetch(url, params):
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2**attempt
            print(f"  {resp.status_code} — retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {resp.status_code}")


def get_count(start, end):
    resp = fetch(f"{BASE}/count", {"starttime": fmt(start), "endtime": fmt(end)})
    time.sleep(COURTESY_DELAY)
    return int(resp.text.strip())


def download(start, end, count=None):
    path = filename(start, end)
    if os.path.exists(path):
        print(f"  skip {os.path.basename(path)} (exists)")
        return

    if count is None:
        count = get_count(start, end)
    print(f"  {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}  count={count}")

    if count == 0:
        return

    if count > LIMIT:
        if (end - start) <= MIN_WINDOW:
            raise RuntimeError(
                f"{count} events in {end - start} — cannot split further"
            )
        mid = start + (end - start) / 2
        download(start, mid)
        download(mid, end)
        return

    resp = fetch(
        f"{BASE}/query",
        {
            "format": "csv",
            "starttime": fmt(start),
            "endtime": fmt(end),
            "orderby": "time-asc",
        },
    )
    time.sleep(COURTESY_DELAY)

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(resp.text)
    os.rename(tmp, path)

    rows = resp.text.count("\n") - 1
    print(f"  wrote {os.path.basename(path)}  ({rows} rows)")


def parse_window(name):
    base = name.removesuffix(".csv")
    s, e = base.split("-")
    return (
        dt.datetime.strptime(s, "%Y%m%d_%H%M"),
        dt.datetime.strptime(e, "%Y%m%d_%H%M"),
    )


def verify(expected):
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    total_rows = 0
    mismatches = []
    for f in files:
        path = os.path.join(RAW_DIR, f)
        with open(path) as fh:
            rows = sum(1 for _ in fh) - 1
        total_rows += rows
        start, end = parse_window(f)
        api_count = get_count(start, end)
        if rows != api_count:
            mismatches.append((f, rows, api_count))
    diff = total_rows - expected
    status = "OK" if diff == 0 and not mismatches else "MISMATCH"
    print(f"\nVerification: {len(files)} files, {total_rows:,} rows on disk, {expected:,} expected — {status}")
    for f, rows, api_count in mismatches:
        print(f"  {f}: {rows} on disk, {api_count} from API ({api_count - rows:+})")
    seen = {}
    dupes = []
    for f in files:
        path = os.path.join(RAW_DIR, f)
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                eid = row["id"]
                if eid in seen:
                    dupes.append((eid, seen[eid], f))
                else:
                    seen[eid] = f
    if dupes:
        print(f"\n  {len(dupes)} duplicate IDs:")
        for eid, first, second in dupes[:10]:
            print(f"    {eid}: {first} and {second}")
        if len(dupes) > 10:
            print(f"    ... and {len(dupes) - 10} more")

    if mismatches or dupes:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Seed USGS earthquake CSVs")
    parser.add_argument("--days", type=int, default=0, help="Seed only the last N days (0 = full history)")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    end = dt.datetime.now(dt.UTC).replace(tzinfo=None) - CUTOFF_LAG
    end = end.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(days=args.days) if args.days else dt.datetime(2015, 1, 1)

    print(f"Seed range: {start:%Y-%m-%d} → {end:%Y-%m-%d}")
    print(f"Cutoff: now minus {CUTOFF_LAG.days} days\n")

    total = 0
    cursor = start
    while cursor < end:
        window_end = min(cursor + dt.timedelta(days=30), end)
        c = get_count(cursor, window_end)
        total += c
        download(cursor, window_end, count=c)
        cursor = window_end

    print(f"\nTotal events in range: {total:,}")

    verify(total)


if __name__ == "__main__":
    main()
