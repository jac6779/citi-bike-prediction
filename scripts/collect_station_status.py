#!/usr/bin/env python3
"""
Collect Citi Bike station_status snapshots and save them to disk.

Designed for local scheduled runs now (Task Scheduler / cron),
and easy to adapt later for AWS Lambda + S3.

Usage examples:
    python collect_station_status.py
    python collect_station_status.py --output-dir data/raw/station_status_snapshots
    python collect_station_status.py --format parquet --compress snappy
    python collect_station_status.py --also-save-json

Notes:
- This script intentionally saves the full station_status payload for schema resilience.
- It adds a collection timestamp (`collected_at_utc`) to every row.
- It writes one file per run, which makes downstream historical reconstruction easy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GBFS_INDEX_URL = "https://gbfs.citibikenyc.com/gbfs/gbfs.json"
DEFAULT_STATION_STATUS_URL = "https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_status.json"
DEFAULT_TIMEOUT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Citi Bike station_status snapshots and save one file per run."
    )
    parser.add_argument(
        "--station-status-url",
        default=DEFAULT_STATION_STATUS_URL,
        help="station_status endpoint to pull from",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/station_status_snapshots",
        help="directory where snapshot files will be written",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="output file format",
    )
    parser.add_argument(
        "--compress",
        default="snappy",
        help="compression codec for parquet output (default: snappy)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--also-save-json",
        action="store_true",
        help="also save the raw JSON payload alongside the tabular snapshot",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def fetch_station_status(url: str, timeout: int) -> dict[str, Any]:
    logging.info("Requesting station_status feed")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Unexpected response type; expected a JSON object.")

    data = payload.get("data", {})
    stations = data.get("stations")
    if not isinstance(stations, list):
        raise ValueError("Could not find station records at payload['data']['stations'].")

    logging.info("Fetched %s station records", len(stations))
    return payload


def build_snapshot_dataframe(payload: dict[str, Any], collected_at: datetime) -> pd.DataFrame:
    stations = payload["data"]["stations"]
    df = pd.DataFrame(stations)

    # Add collection metadata to every row.
    df["collected_at_utc"] = pd.Timestamp(collected_at)
    df["source_url"] = DEFAULT_STATION_STATUS_URL
    df["gbfs_last_updated_unix"] = payload.get("last_updated")
    df["gbfs_ttl_seconds"] = payload.get("ttl")

    # Helpful parsed datetime for the feed-level last_updated field.
    if "gbfs_last_updated_unix" in df.columns:
        df["gbfs_last_updated_utc"] = pd.to_datetime(
            df["gbfs_last_updated_unix"], unit="s", utc=True, errors="coerce"
        )

    return df


def make_base_filename(collected_at: datetime) -> str:
    return collected_at.strftime("station_status_%Y-%m-%d_%H-%M-%S")


def save_snapshot(
    df: pd.DataFrame,
    output_dir: Path,
    base_filename: str,
    file_format: str,
    compress: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_format == "parquet":
        outpath = output_dir / f"{base_filename}.parquet"
        df.to_parquet(outpath, index=False, compression=compress)
    else:
        outpath = output_dir / f"{base_filename}.csv"
        df.to_csv(outpath, index=False)

    return outpath


def save_raw_json(payload: dict[str, Any], output_dir: Path, base_filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / f"{base_filename}.json"
    with outpath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return outpath


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    collected_at = datetime.now(timezone.utc)
    output_dir = Path(args.output_dir)
    base_filename = make_base_filename(collected_at)

    try:
        payload = fetch_station_status(args.station_status_url, args.timeout)
        df = build_snapshot_dataframe(payload, collected_at)

        snapshot_path = save_snapshot(
            df=df,
            output_dir=output_dir,
            base_filename=base_filename,
            file_format=args.format,
            compress=args.compress,
        )
        logging.info("Saved tabular snapshot to %s", snapshot_path)

        if args.also_save_json:
            json_dir = output_dir / "raw_json"
            json_path = save_raw_json(payload, json_dir, base_filename)
            logging.info("Saved raw JSON payload to %s", json_path)

        logging.info("Done")
        return 0

    except requests.HTTPError as exc:
        logging.exception("HTTP error while pulling station_status: %s", exc)
        return 1
    except requests.RequestException as exc:
        logging.exception("Network error while pulling station_status: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.exception("Collector failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
