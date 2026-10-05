"""Data ingestion and processing pipeline for Smart Facility IoT sensor streams.

Extracts, cleans, resamples, and aligns environmental telemetry (5 sensors, 180k rows)
and room occupancy event telemetry (5 zones, 9.1M rows) from Monash Smart Building network.

Outputs:
    - Cleaned, resampled, and aligned 5-minute time series exported as Parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_pipeline")

# The 5 environmental sensors in 5EnvSensor
TARGET_ENV_SENSOR_IDS = {
    "6012002000241",
    "6012002000326",
    "6012002000869",
    "6012002000227",
    "6012002000777",
}

# The 5 occupancy spaces in 5occupancySensor
TARGET_FLOORSPACE_IDS = {
    "0f3f37f8-0335-41e0-ae87-a1d677dfc7d7",
    "941f0352-bb69-47d0-a9c5-ac29eb4f7319",
    "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b",
    "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8",
    "227f1068-9e41-469f-bef0-275f27a0bdba",
}


def parse_env_json_payload(payload_str: str) -> List[Dict[str, Any]]:
    """Parse JSON measurement payload from environmental sensor stream."""
    if not isinstance(payload_str, str) or not payload_str.strip():
        return []

    try:
        data = json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        return []

    records = []
    for item in data:
        var_info = item.get("variable", {})
        var_name = var_info.get("name")
        values = item.get("values", [])

        for val_entry in values:
            t = val_entry.get("time")
            v = val_entry.get("value")
            err = val_entry.get("errorCode", 0)
            if t is not None and v is not None and err == 0:
                records.append({
                    "time": t,
                    "variable": var_name,
                    "value": float(v),
                })
    return records


def extract_environmental_records(
    csv_path: str | Path,
    target_sensors: Optional[Set[str]] = None,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Stream and parse environmental telemetry CSV.

    Args:
        csv_path: Path to 5EnvSensor CSV.
        target_sensors: Optional set of sensor IDs to filter. Defaults to TARGET_ENV_SENSOR_IDS.
        chunk_size: CSV streaming chunk size.

    Returns:
        DataFrame with aggregated environmental measurements.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Environmental CSV not found: {csv_path}")

    target_sensors = target_sensors or TARGET_ENV_SENSOR_IDS
    logger.info("Extracting environmental telemetry from %s...", csv_path.name)
    extracted_rows: List[Dict[str, Any]] = []

    var_mapping = {
        "Carbon dioxide": "co2_ppm",
        "Temperature": "temp_c",
        "Humidity": "humidity_rh",
        "Particulate matter 2.5": "pm25_ugm3",
        "Particulate matter 10": "pm10_ugm3",
        "Particulate matter 1": "pm1_ugm3",
        "Particulate matter 4": "pm4_ugm3",
        "Light Volatile Organic Compounds": "voc_ppb",
        "Formaldehyde": "formaldehyde_ugm3",
        "Pressure": "pressure_mb",
    }

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str, on_bad_lines="skip"):
        chunk.columns = chunk.columns.str.lower()
        sensor_col = "sensorid" if "sensorid" in chunk.columns else chunk.columns[1]
        data_col = "jsondata" if "jsondata" in chunk.columns else chunk.columns[3]

        mask = chunk[sensor_col].astype(str).str.strip().isin(target_sensors)
        matched = chunk[mask]

        for sid, raw_json in zip(matched[sensor_col], matched[data_col]):
            parsed = parse_env_json_payload(str(raw_json))
            for p in parsed:
                v_name = p["variable"]
                if v_name in var_mapping:
                    extracted_rows.append({
                        "timestamp": pd.to_datetime(p["time"], unit="s", utc=True),
                        "sensor_id": str(sid).strip(),
                        "metric": var_mapping[v_name],
                        "value": p["value"],
                    })

    logger.info("Extracted %d valid measurements across target environmental sensors.", len(extracted_rows))
    if not extracted_rows:
        return pd.DataFrame()

    df_raw = pd.DataFrame(extracted_rows)

    # Pivot so metrics are columns
    df_pivot = df_raw.pivot_table(
        index=["timestamp", "sensor_id"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Aggregate across sensors to compute mean facility state
    cols_to_mean = [c for c in var_mapping.values() if c in df_pivot.columns]
    df_room = df_pivot.groupby("timestamp")[cols_to_mean].mean().reset_index()

    return df_room


def extract_occupancy_records(
    csv_path: str | Path,
    target_spaces: Optional[Set[str]] = None,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    """Stream and filter 9M row occupancy CSV for target spaces.

    Args:
        csv_path: Path to 5occupancySensor CSV.
        target_spaces: Optional set of space IDs. Defaults to TARGET_FLOORSPACE_IDS.
        chunk_size: Streaming chunk size.

    Returns:
        DataFrame of timestamped headcount observations.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Occupancy CSV not found: {csv_path}")

    target_spaces = target_spaces or TARGET_FLOORSPACE_IDS
    logger.info("Streaming and aggregating occupancy telemetry from %s...", csv_path.name)

    cols_to_use = ["floorspaceid", "headcount", "collecteddate"]
    matched_chunks: List[pd.DataFrame] = []
    total_processed = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=cols_to_use, dtype=str, on_bad_lines="skip"):
        total_processed += len(chunk)
        mask = chunk["floorspaceid"].str.strip().isin(target_spaces)
        if mask.any():
            sub = chunk[mask].copy()
            matched_chunks.append(sub)

        if total_processed % 2_000_000 == 0:
            logger.info("Processed %d / 9.1M rows...", total_processed)

    logger.info("Completed scanning %d occupancy rows.", total_processed)
    if not matched_chunks:
        return pd.DataFrame()

    df_occ = pd.concat(matched_chunks, ignore_index=True)
    df_occ["headcount"] = pd.to_numeric(df_occ["headcount"], errors="coerce").fillna(0.0)
    df_occ["timestamp"] = pd.to_datetime(df_occ["collecteddate"], utc=True)

    # Sort chronologically
    df_occ = df_occ.sort_values("timestamp").reset_index(drop=True)
    return df_occ[["timestamp", "floorspaceid", "headcount"]]


def clean_and_resample(
    df_env: pd.DataFrame,
    df_occ: pd.DataFrame,
    resample_freq: str = "5min",
) -> pd.DataFrame:
    """Resample environmental and occupancy streams to uniform time grid and merge."""
    logger.info("Resampling time series to uniform frequency %s...", resample_freq)

    # 1. Environmental resampling
    df_env = df_env.set_index("timestamp").sort_index()
    env_resampled = df_env.resample(resample_freq).median()

    # Physical bounds filtering
    if "co2_ppm" in env_resampled.columns:
        env_resampled.loc[(env_resampled["co2_ppm"] < 350) | (env_resampled["co2_ppm"] > 5000), "co2_ppm"] = np.nan
    if "temp_c" in env_resampled.columns:
        env_resampled.loc[(env_resampled["temp_c"] < 10) | (env_resampled["temp_c"] > 45), "temp_c"] = np.nan
    if "humidity_rh" in env_resampled.columns:
        env_resampled.loc[(env_resampled["humidity_rh"] < 5) | (env_resampled["humidity_rh"] > 95), "humidity_rh"] = np.nan

    # Interpolate short gaps (up to 30 min) linearly
    env_resampled = env_resampled.interpolate(method="time", limit=6)

    # 2. Occupancy resampling
    # For each space, compute the max headcount per 5-min window, then sum across spaces for total facility occupancy
    df_occ = df_occ.sort_values("timestamp")
    occ_by_space = df_occ.groupby(["floorspaceid", pd.Grouper(key="timestamp", freq=resample_freq)])["headcount"].max().reset_index()
    occ_total = occ_by_space.groupby("timestamp")["headcount"].sum().to_frame()

    occ_total["is_occupied"] = (occ_total["headcount"] > 0).astype(int)

    # 3. Inner join on overlapping timestamp range
    merged = env_resampled.join(occ_total, how="inner").reset_index()

    # Drop any rows where core features remain null
    merged = merged.dropna(subset=["co2_ppm", "temp_c", "headcount"]).reset_index(drop=True)
    logger.info("Successfully produced %d synchronized 5-minute intervals.", len(merged))

    return merged


def run_pipeline(
    env_csv: str | Path,
    occ_csv: str | Path,
    output_parquet: str | Path,
    output_summary: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Execute complete data pipeline and export results."""
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df_env = extract_environmental_records(env_csv)
    df_occ = extract_occupancy_records(occ_csv)

    if df_env.empty or df_occ.empty:
        raise ValueError("Failed to extract data: one or both streams returned 0 rows.")

    df_merged = clean_and_resample(df_env, df_occ, resample_freq="5min")

    df_merged.to_parquet(output_parquet, index=False)
    logger.info("Saved merged dataset to %s (%d rows)", output_parquet, len(df_merged))

    if output_summary:
        summary_path = Path(output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_info = {
            "total_rows": len(df_merged),
            "start_time": str(df_merged["timestamp"].min()),
            "end_time": str(df_merged["timestamp"].max()),
            "features": list(df_merged.columns),
            "mean_co2": float(df_merged["co2_ppm"].mean()),
            "max_co2": float(df_merged["co2_ppm"].max()),
            "mean_temp": float(df_merged["temp_c"].mean()),
            "mean_headcount": float(df_merged["headcount"].mean()),
            "max_headcount": float(df_merged["headcount"].max()),
            "occupied_fraction": float(df_merged["is_occupied"].mean()),
        }
        with open(summary_path, "w") as f:
            json.dump(summary_info, f, indent=2)
        logger.info("Saved data summary metadata to %s", summary_path)

    return df_merged


def main():
    parser = argparse.ArgumentParser(description="Ingest, clean, and align IoT telemetry.")
    parser.add_argument("--env-csv", required=True, help="Path to 5EnvSensor CSV.")
    parser.add_argument("--occ-csv", required=True, help="Path to 5occupancySensor CSV.")
    parser.add_argument("--output", default=".tmp/g57_merged_data.parquet", help="Output Parquet path.")
    parser.add_argument("--summary", default=".tmp/data_summary.json", help="Summary JSON output path.")

    args = parser.parse_args()
    run_pipeline(args.env_csv, args.occ_csv, args.output, args.summary)


if __name__ == "__main__":
    main()
