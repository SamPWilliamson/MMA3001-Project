"""Validation, out-of-distribution verification, and computational profiling suite.

Measures runtime latency, memory allocations, and theoretical arithmetic complexity (FLOPs),
and executes domain envelope validation against out-of-distribution input queries.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from execution.models import BaselineKNN, DecisionTreeModel, GaussianProcessModel, RandomForestModel
from execution.numerical_methods import OutOfDistributionDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("profiling")


def profile_model_latencies(
    models_dict: Dict[str, Any],
    X_sample: np.ndarray,
    n_repetitions: int = 500,
) -> Dict[str, Dict[str, float]]:
    """Benchmark query inference latency and memory usage.

    Args:
        models_dict: Dictionary of model names to fitted instances.
        X_sample: 2D array of query samples.
        n_repetitions: Number of iterations for timing measurement.

    Returns:
        Dictionary of latency (microseconds per sample) and memory (KB).
    """
    logger.info("Profiling inference latency and memory over %d repetitions...", n_repetitions)
    benchmarks: Dict[str, Dict[str, float]] = {}

    single_query = X_sample[:1]

    for name, model in models_dict.items():
        # Memory tracking
        tracemalloc.start()
        _ = model.predict(X_sample[:50])
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Timing single-sample query latency
        t0 = time.perf_counter()
        for _ in range(n_repetitions):
            _ = model.predict(single_query)
        elapsed = time.perf_counter() - t0
        latency_us = (elapsed / n_repetitions) * 1e6

        benchmarks[name] = {
            "latency_us_per_query": float(latency_us),
            "peak_memory_kb": float(peak_mem / 1024.0),
        }

    return benchmarks


def estimate_theoretical_complexity(
    n_train: int,
    n_features: int,
    tree_depth: int = 5,
    n_trees: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Calculate theoretical arithmetic complexity and FLOP estimates per query.

    Args:
        n_train: Number of training exemplars.
        n_features: Dimensionality D.
        tree_depth: Average tree depth.
        n_trees: Number of trees in ensemble.

    Returns:
        Dictionary containing Big-O notation and estimated FLOP count per prediction.
    """
    # 1. KNN: Distance computation between query and N training points -> (3*D) operations per point
    knn_flops = n_train * (3 * n_features) + n_train  # Distance calc + partial sort
    # 2. Decision Tree: depth comparisons
    dt_ops = tree_depth  # purely scalar comparisons
    # 3. Gaussian Process: Kernel computation against N training points + dot product
    # RBF kernel: exp(-sum((x - x_i)^2)/(2*l^2)) -> ~5*D + 5 ops per point, then vector dot product (2*N)
    gp_flops = n_train * (5 * n_features + 5) + 2 * n_train
    # 4. Random Forest: n_trees * depth
    rf_ops = n_trees * tree_depth

    return {
        "Baseline KNN": {
            "complexity": "O(N * D)",
            "estimated_flops_per_query": int(knn_flops),
            "nature": "Distance matrix evaluation",
        },
        "Decision Tree": {
            "complexity": "O(depth)",
            "estimated_flops_per_query": int(dt_ops),
            "nature": "Hierarchical threshold traversal",
        },
        "Gaussian Process": {
            "complexity": "O(N * D)",
            "estimated_flops_per_query": int(gp_flops),
            "nature": "Nonlinear kernel vector dot product",
        },
        "Random Forest": {
            "complexity": "O(T * depth)",
            "estimated_flops_per_query": int(rf_ops),
            "nature": "Ensemble decision routing",
        },
    }


def verify_out_of_distribution_behavior(
    detector: OutOfDistributionDetector,
) -> List[Dict[str, Any]]:
    """Test response of OOD envelope detector on synthetic benchmark queries.

    Verifies requirement: code indicates when a prediction falls outside what
    the data supports.
    """
    logger.info("Executing Out-of-Distribution (OOD) test suite...")
    test_cases = [
        {
            "description": "Nominal Weekday Afternoon Reading",
            "query_co2": 550.0,
            "query_temp": 22.5,
            "query_hour": 14.0,
            "expected_support": True,
        },
        {
            "description": "Extreme CO2 Surge (Faulty Sensor or Extreme Crowding)",
            "query_co2": 3200.0,
            "query_temp": 24.0,
            "query_hour": 15.0,
            "expected_support": False,
        },
        {
            "description": "HVAC Thermal Failure (Excessive Temperature)",
            "query_co2": 700.0,
            "query_temp": 46.0,
            "query_hour": 13.0,
            "expected_support": False,
        },
        {
            "description": "Impossible Sub-Ambient CO2 Reading (Negative / Below 350 ppm)",
            "query_co2": 280.0,
            "query_temp": 20.0,
            "query_hour": 11.0,
            "expected_support": False,
        },
    ]

    results = []
    for tc in test_cases:
        # Construct synthetic query vector matching features
        # (Assuming order: co2, temp, hour)
        q = np.array([tc["query_co2"], tc["query_temp"], tc["query_hour"]])
        val_res = detector.validate_query(q)
        results.append({
            "test_case": tc["description"],
            "inputs": {
                "co2_ppm": tc["query_co2"],
                "temp_c": tc["query_temp"],
                "hour": tc["query_hour"],
            },
            "supported": val_res["supported"],
            "confidence": val_res["confidence"],
            "flags": val_res["flags"],
            "matches_expectation": val_res["supported"] == tc["expected_support"],
        })

    return results


def run_full_validation_suite(
    features_parquet: str | Path,
    output_validation_json: str | Path,
    output_profiling_txt: str | Path,
) -> None:
    """Run profiling, theoretical complexity analysis, and OOD tests."""
    df = pd.read_parquet(features_parquet)

    # Prepare features
    features_base = ["co2_ppm", "time_of_day_hours"]
    features_full = [
        "co2_ppm", "co2_rate_ppm_min", "co2_above_ambient", "co2_exposure_integral_3h",
        "temp_c", "temp_rate_c_min", "humidity_rh", "humidity_rate_rh_min",
        "time_of_day_hours", "day_of_week", "is_operating_hours", "is_weekend",
        "sin_hour", "cos_hour", "sin_day", "cos_day",
    ]
    features_full = [f for f in features_full if f in df.columns]

    X_base = df[features_base].values
    X_full = df[features_full].values
    y = df["headcount"].values

    # Fit lightweight models for profiling
    n_fit = min(2000, len(X_full))
    knn = BaselineKNN(n_neighbors=5).fit(X_base[:n_fit], y[:n_fit])
    dt = DecisionTreeModel(max_depth=5).fit(X_full[:n_fit], y[:n_fit])
    gpr = GaussianProcessModel(max_train_samples=1000).fit(X_full[:n_fit], y[:n_fit])
    rf = RandomForestModel(n_estimators=50, max_depth=10).fit(X_full[:n_fit], y[:n_fit])

    # Benchmarks
    lat_knn = profile_model_latencies({"Baseline KNN": knn}, X_base[:100])
    lat_others = profile_model_latencies({
        "Decision Tree": dt,
        "Gaussian Process": gpr,
        "Random Forest": rf,
    }, X_full[:100])
    latency_results = {**lat_knn, **lat_others}

    complexity_results = estimate_theoretical_complexity(
        n_train=n_fit,
        n_features=len(features_full),
    )

    # OOD detector on 3 key features: CO2, Temp, Hour
    ood_data = df[["co2_ppm", "temp_c", "time_of_day_hours"]].values
    detector = OutOfDistributionDetector(
        ood_data,
        feature_names=["CO2 (ppm)", "Temperature (C)", "Time of Day (hr)"],
        std_multiplier=6.0,
    )
    ood_results = verify_out_of_distribution_behavior(detector)

    # Save outputs
    val_payload = {
        "out_of_distribution_tests": ood_results,
        "theoretical_complexity": complexity_results,
        "empirical_benchmarks": latency_results,
    }

    out_val = Path(output_validation_json)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    with open(out_val, "w") as f:
        json.dump(val_payload, f, indent=2)
    logger.info("Saved validation report to %s", out_val)

    # Generate readable text summary
    out_prof = Path(output_profiling_txt)
    with open(out_prof, "w") as f:
        f.write("======================================================================\n")
        f.write("MMA3001 COMPUTATIONAL PROFILING AND BENCHMARKING REPORT\n")
        f.write("======================================================================\n\n")
        f.write("1. EMPIRICAL INFERENCE LATENCY & MEMORY (per query):\n")
        f.write(f"{'Model':<25} | {'Latency (us)':<15} | {'Peak Memory (KB)':<15}\n")
        f.write("-" * 60 + "\n")
        for m_name, res in latency_results.items():
            f.write(f"{m_name:<25} | {res['latency_us_per_query']:<15.2f} | {res['peak_memory_kb']:<15.2f}\n")

        f.write("\n2. THEORETICAL ARITHMETIC COMPLEXITY & FLOP ESTIMATES:\n")
        f.write(f"{'Model':<25} | {'Complexity':<12} | {'Estimated FLOPs':<18} | {'Operation Type'}\n")
        f.write("-" * 75 + "\n")
        for m_name, comp in complexity_results.items():
            f.write(f"{m_name:<25} | {comp['complexity']:<12} | {comp['estimated_flops_per_query']:<18} | {comp['nature']}\n")

        f.write("\n3. OUT-OF-DISTRIBUTION ENVELOPE VERIFICATION:\n")
        for r in ood_results:
            status = "PASSED" if r["matches_expectation"] else "FAILED"
            f.write(f"[{status}] {r['test_case']} -> Supported: {r['supported']}, Confidence: {r['confidence']:.2f}\n")
            if r["flags"]:
                f.write(f"        Flags: {', '.join(r['flags'])}\n")

    logger.info("Saved profiling report text to %s", out_prof)


def main():
    parser = argparse.ArgumentParser(description="Profile model latency and verify OOD envelope.")
    parser.add_argument("--features", default=".tmp/g57_features_matrix.parquet", help="Feature parquet path.")
    parser.add_argument("--output-val", default=".tmp/validation_report.json", help="Validation JSON output.")
    parser.add_argument("--output-prof", default=".tmp/profiling_report.txt", help="Profiling report TXT output.")

    args = parser.parse_args()
    run_full_validation_suite(args.features, args.output_val, args.output_prof)


if __name__ == "__main__":
    main()
