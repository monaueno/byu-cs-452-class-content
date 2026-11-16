"""Merge raw CSVs into a deduplicated seed.parquet using DuckDB."""

import os
import duckdb

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PARQUET = os.path.join(os.path.dirname(__file__), "seed.parquet")


def main():
    db = duckdb.connect()
    db.execute("SET TimeZone = 'UTC'")

    db.execute("""
        COPY (
            WITH raw AS (
                SELECT *
                FROM read_csv(?, header=true, types={
                    'time': 'TIMESTAMPTZ',
                    'latitude': 'DOUBLE',
                    'longitude': 'DOUBLE',
                    'depth': 'DOUBLE',
                    'mag': 'DOUBLE',
                    'magType': 'VARCHAR',
                    'nst': 'INTEGER',
                    'gap': 'DOUBLE',
                    'dmin': 'DOUBLE',
                    'rms': 'DOUBLE',
                    'net': 'VARCHAR',
                    'id': 'VARCHAR',
                    'updated': 'TIMESTAMPTZ',
                    'place': 'VARCHAR',
                    'type': 'VARCHAR',
                    'horizontalError': 'DOUBLE',
                    'depthError': 'DOUBLE',
                    'magError': 'DOUBLE',
                    'magNst': 'INTEGER',
                    'status': 'VARCHAR',
                    'locationSource': 'VARCHAR',
                    'magSource': 'VARCHAR'
                })
            ),
            deduped AS (
                SELECT *
                FROM raw
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY id ORDER BY updated DESC
                ) = 1
            )
            SELECT *,
                'seed' AS source,
                NOW() AS ingested_at
            FROM deduped
            ORDER BY time
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
    """, [os.path.join(RAW_DIR, "*.csv"), PARQUET])

    count = db.execute(
        "SELECT COUNT(*) FROM read_parquet(?)", [PARQUET]
    ).fetchone()[0]

    size_mb = os.path.getsize(PARQUET) / 1_048_576
    print(f"seed.parquet: {count:,} rows, {size_mb:.1f} MB\n")

    print("Schema:")
    for row in db.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [PARQUET]
    ).fetchall():
        print(f"  {row[0]:20s} {row[1]}")

    print("\nSample rows:")
    print(db.execute(
        "SELECT time, id, mag, place, source, ingested_at FROM read_parquet(?) LIMIT 5",
        [PARQUET],
    ).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
