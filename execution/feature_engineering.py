"""Feature engineering driver applying syllabus numerical methods to IoT streams.

This module processes clean resampled environmental and occupancy data,
applying numerical differentiation, numerical integration, and cyclical time encodings
to generate a high-dimensional predictive feature matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from execution.numerical_methods import central_difference_derivative, cumulative_trapezoidal_rule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("feature_engineering")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw synchronized sensor telemetry into predictive feature matrix.

    Args:
        df: Synchronized DataFrame containing timestamp, environmental metrics, and occupancy.

    Returns:
        Augmented DataFrame with numerical derivatives, integrals, and temporal features.
    """
    logger.info("Starting feature engineering on %d records...", len(df))
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Temporal context features
    ts = pd.to_datetime(df["timestamp"])
    hours = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    day_of_week = ts.dt.dayofweek  # 0 = Monday, 6 = Sunday

    df["time_of_day_hours"] = hours
    df["day_of_week"] = day_of_week
    df["is_weekend"] = (day_of_week >= 5).astype(int)
    # Monash Building core operating hours: Weekdays 08:00 to 18:00
    df["is_operating_hours"] = ((day_of_week < 5) & (hours >= 8.0) & (hours <= 18.0)).astype(int)

    # Cyclical trigonometric transforms for continuous time representation
    df["sin_hour"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["sin_day"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    df["cos_day"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    # Time delta in minutes for numerical schemes (nominal 5.0 minutes)
    time_min = (ts - ts.iloc[0]).dt.total_seconds().values / 60.0

    # 2. Numerical Differentiation (Week 7 finite difference scheme)
    # Pre-smoothed central differences prevent high-frequency noise amplification
    if "co2_ppm" in df.columns:
        co2_vals = df["co2_ppm"].values
        df["co2_rate_ppm_min"] = central_difference_derivative(time_min, co2_vals, smooth_window=5)
        # Baseline offset above ambient outdoor air (~415 ppm)
        df["co2_above_ambient"] = np.maximum(co2_vals - 415.0, 0.0)

    if "temp_c" in df.columns:
        temp_vals = df["temp_c"].values
        df["temp_rate_c_min"] = central_difference_derivative(time_min, temp_vals, smooth_window=5)

    if "humidity_rh" in df.columns:
        rh_vals = df["humidity_rh"].values
        df["humidity_rate_rh_min"] = central_difference_derivative(time_min, rh_vals, smooth_window=5)

    # 3. Numerical Integration (Week 6 running exposure / air renewal deficit)
    # Cumulative CO2 excess over past 3 hours (36 five-minute intervals)
    if "co2_above_ambient" in df.columns:
        # Rolling 3-hour cumulative numerical integral (composite trapezoidal)
        rolling_window = 36
        co2_excess = df["co2_above_ambient"].values
        # Running integral over a sliding window
        running_integral = np.zeros(len(df))
        for i in range(len(df)):
            start_idx = max(0, i - rolling_window)
            if i - start_idx >= 2:
                sub_y = co2_excess[start_idx : i + 1]
                # dt in hours (5 min = 5/60 = 1/12 hr)
                running_integral[i] = cumulative_trapezoidal_rule(sub_y, dx=5.0 / 60.0)[-1]
        df["co2_exposure_integral_3h"] = running_integral

    # 4. Multi-pollutant particulate ratio
    if "pm25_ugm3" in df.columns and "pm10_ugm3" in df.columns:
        df["pm_ratio_fine_to_coarse"] = df["pm25_ugm3"] / (df["pm10_ugm3"] + 1e-3)

    logger.info("Feature engineering complete. Produced %d feature columns.", len(df.columns))
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate numerical features from merged sensor data.")
    parser.add_argument("--input", default=".tmp/g57_merged_data.parquet", help="Input parquet path.")
    parser.add_argument("--output", default=".tmp/g57_features_matrix.parquet", help="Output parquet path.")
    parser.add_argument("--manifest", default=".tmp/feature_manifest.json", help="Output feature manifest JSON.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_in = pd.read_parquet(input_path)
    df_out = engineer_features(df_in)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    logger.info("Saved feature matrix to %s", output_path)

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "total_records": len(df_out),
            "columns": list(df_out.columns),
            "numerical_features": [
                "co2_ppm", "co2_rate_ppm_min", "co2_above_ambient", "co2_exposure_integral_3h",
                "temp_c", "temp_rate_c_min", "humidity_rh", "humidity_rate_rh_min",
                "time_of_day_hours", "day_of_week", "is_operating_hours", "is_weekend",
                "sin_hour", "cos_hour", "sin_day", "cos_day",
            ],
            "targets": ["headcount", "is_occupied"],
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
