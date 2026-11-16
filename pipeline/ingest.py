"""Hourly live-ingest: fetch new USGS earthquakes, append to BigQuery."""

import argparse
import io
import os
import time
import datetime as dt
import requests
import pandas as pd
from google.cloud import bigquery

BASE = "https://earthquake.usgs.gov/fdsnws/event/1"
PROJECT = os.environ["GCP_PROJECT"]
TABLE = f"{PROJECT}.earthquakes.events"
PAGE_LIMIT = 20_000
REQUEST_TIMEOUT = 120
COURTESY_DELAY = 0.5
MAX_RETRIES = 5

BQ_SCHEMA = [
    bigquery.SchemaField("time", "TIMESTAMP"),
    bigquery.SchemaField("latitude", "FLOAT64"),
    bigquery.SchemaField("longitude", "FLOAT64"),
    bigquery.SchemaField("depth", "FLOAT64"),
    bigquery.SchemaField("mag", "FLOAT64"),
    bigquery.SchemaField("magType", "STRING"),
    bigquery.SchemaField("nst", "INT64"),
    bigquery.SchemaField("gap", "FLOAT64"),
    bigquery.SchemaField("dmin", "FLOAT64"),
    bigquery.SchemaField("rms", "FLOAT64"),
    bigquery.SchemaField("net", "STRING"),
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("updated", "TIMESTAMP"),
    bigquery.SchemaField("place", "STRING"),
    bigquery.SchemaField("type", "STRING"),
    bigquery.SchemaField("horizontalError", "FLOAT64"),
    bigquery.SchemaField("depthError", "FLOAT64"),
    bigquery.SchemaField("magError", "FLOAT64"),
    bigquery.SchemaField("magNst", "INT64"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("locationSource", "STRING"),
    bigquery.SchemaField("magSource", "STRING"),
    bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
]


def fmt(d):
    return d.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_http(url, params):
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (200, 204, 404):
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2**attempt
            print(f"  {resp.status_code} — retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {resp.status_code}")


def get_high_water(bq):
    row = next(bq.query(f"""
        SELECT
            MAX(IF(source = 'live', updated, NULL)) AS live_hw,
            MAX(IF(source = 'seed', time, NULL)) AS seed_hw
        FROM `{TABLE}`
    """).result())

    if row.live_hw is not None:
        start = row.live_hw - dt.timedelta(hours=1)
        mode = "live"
        print(f"Mode: live (last live updated = {row.live_hw:%Y-%m-%d %H:%M} UTC)")
    elif row.seed_hw is not None:
        start = row.seed_hw - dt.timedelta(hours=1)
        mode = "first"
        print(f"Mode: first run (seed ends at {row.seed_hw:%Y-%m-%d %H:%M} UTC)")
    else:
        start = dt.datetime(2015, 1, 1, tzinfo=dt.UTC)
        mode = "first"
        print("Mode: empty table")

    return start, mode


def get_count(params):
    resp = fetch_http(f"{BASE}/count", params)
    time.sleep(COURTESY_DELAY)
    if resp.status_code in (204, 404):
        return 0
    return int(resp.text.strip())


def fetch_events(start, mode):
    if mode == "first":
        params = {"format": "csv", "starttime": fmt(start), "orderby": "time-asc"}
    else:
        params = {"format": "csv", "updatedafter": fmt(start), "orderby": "time-asc"}

    count_params = {k: v for k, v in params.items() if k != "format"}
    total = get_count(count_params)
    print(f"USGS reports {total:,} events")

    if total == 0:
        return ""

    if total <= PAGE_LIMIT:
        resp = fetch_http(f"{BASE}/query", params)
        time.sleep(COURTESY_DELAY)
        if resp.status_code in (204, 404):
            return ""
        return resp.text

    all_lines = []
    header = None
    offset = 1
    while offset <= total:
        page_params = {**params, "limit": PAGE_LIMIT, "offset": offset}
        resp = fetch_http(f"{BASE}/query", page_params)
        time.sleep(COURTESY_DELAY)
        if resp.status_code in (204, 404):
            break
        lines = resp.text.splitlines()
        if header is None:
            header = lines[0]
            all_lines.append(header)
        all_lines.extend(lines[1:])
        fetched = len(lines) - 1
        print(f"  page offset={offset}: {fetched} rows")
        if fetched < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT

    return "\n".join(all_lines) + "\n" if all_lines else ""


DTYPE_MAP = {
    "FLOAT64": "Float64",
    "INT64": "Int64",
    "STRING": "object",
    "TIMESTAMP": "datetime64[us, UTC]",
}


def parse_rows(csv_text):
    df = pd.read_csv(io.StringIO(csv_text))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["updated"] = pd.to_datetime(df["updated"], utc=True)
    df["source"] = "live"
    df["ingested_at"] = dt.datetime.now(dt.UTC)

    for field in BQ_SCHEMA:
        expected_dtype = DTYPE_MAP[field.field_type]
        if field.name not in df.columns:
            raise ValueError(f"Missing column: {field.name}")
        df[field.name] = df[field.name].astype(expected_dtype)

    df = df[[f.name for f in BQ_SCHEMA]]
    return df


def append_rows(bq, df):
    job_config = bigquery.LoadJobConfig(
        schema=BQ_SCHEMA,
        write_disposition="WRITE_APPEND",
    )
    job = bq.load_table_from_dataframe(df, TABLE, job_config=job_config)
    job.result()
    print(f"Loaded {len(df)} rows (job {job.job_id})")


def main():
    parser = argparse.ArgumentParser(description="USGS earthquake live ingest")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and validate but do not load")
    args = parser.parse_args()

    bq = bigquery.Client(project=PROJECT)

    start, mode = get_high_water(bq)
    print(f"Watermark: {start:%Y-%m-%d %H:%M} UTC")

    csv_text = fetch_events(start, mode)
    if not csv_text or csv_text.count("\n") <= 1:
        print("Rows fetched: 0")
        print("Rows loaded: 0")
        return

    df = parse_rows(csv_text)
    print(f"Rows fetched: {len(df)}")

    if args.dry_run:
        print("Rows loaded: 0 (dry run)")
        return

    append_rows(bq, df)
    print(f"Rows loaded: {len(df)}")


if __name__ == "__main__":
    main()
