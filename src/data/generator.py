"""
Synthetic Manufacturing Data Generator

Generates realistic batch manufacturing data with:
- Multiple products, sites, process steps
- Correlated metrics
- Time-based drift
- Occasional events (bad lots, maintenance)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Configuration
PRODUCTS = ["BioProduct-A", "BioProduct-B", "CellTherapy-X", "GeneVector-Z"]
SITES = ["Site-Boston", "Site-Dublin", "Site-Singapore"]
LINES = {
    "Site-Boston": ["Line-1", "Line-2"],
    "Site-Dublin": ["Line-1"],
    "Site-Singapore": ["Line-1", "Line-2", "Line-3"]
}
PROCESS_STEPS = [
    "Thaw & Seed",
    "Expansion Day 3",
    "Expansion Day 7",
    "Harvest",
    "Formulation",
    "Fill & Finish",
    "QC Release"
]
SHIFTS = ["Day", "Evening", "Night"]
OPERATORS = [f"OP-{i:03d}" for i in range(1, 21)]

# Specification limits for key metrics
SPEC_LIMITS = {
    "yield_pct": {"LSL": 60.0, "USL": 100.0, "target": 85.0},
    "viability_pct": {"LSL": 70.0, "USL": 100.0, "target": 92.0},
    "potency_proxy": {"LSL": 80.0, "USL": 120.0, "target": 100.0},
    "impurity_pct": {"LSL": 0.0, "USL": 5.0, "target": 1.5},
    "cycle_time_hrs": {"LSL": None, "USL": 48.0, "target": 36.0},
    "hold_time_hrs": {"LSL": 0.0, "USL": 24.0, "target": 4.0},
}


def generate_batch_data(
    n_batches: int = 500,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate synthetic batch manufacturing data.

    Parameters:
    -----------
    n_batches : int
        Number of batches to generate
    start_date : str
        Start date for batch generation
    end_date : str
        End date for batch generation
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing batch records with all metrics
    """
    np.random.seed(seed)

    # Generate date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days

    records = []

    for i in range(n_batches):
        # Basic identifiers
        batch_id = f"B-{start.year}-{i+1:05d}"
        lot_id = f"L-{np.random.randint(1, 100):03d}"

        # Random date within range
        days_offset = np.random.randint(0, date_range + 1)
        batch_date = start + timedelta(days=days_offset)

        # Assign product, site, line
        product = np.random.choice(PRODUCTS, p=[0.35, 0.30, 0.20, 0.15])
        site = np.random.choice(SITES, p=[0.40, 0.25, 0.35])
        line = np.random.choice(LINES[site])

        # Shift and operator
        shift = np.random.choice(SHIFTS, p=[0.45, 0.35, 0.20])
        operator = np.random.choice(OPERATORS)

        # Generate process step data
        for step_idx, step in enumerate(PROCESS_STEPS):
            record = generate_step_record(
                batch_id=batch_id,
                lot_id=lot_id,
                batch_date=batch_date,
                product=product,
                site=site,
                line=line,
                shift=shift,
                operator=operator,
                process_step=step,
                step_idx=step_idx,
                batch_idx=i,
                n_batches=n_batches
            )
            records.append(record)

    df = pd.DataFrame(records)

    # Add derived columns
    df = add_derived_metrics(df)

    return df


def generate_step_record(
    batch_id: str,
    lot_id: str,
    batch_date: datetime,
    product: str,
    site: str,
    line: str,
    shift: str,
    operator: str,
    process_step: str,
    step_idx: int,
    batch_idx: int,
    n_batches: int
) -> Dict:
    """Generate a single process step record with realistic metrics."""

    # Base parameters by product (simulate product-specific behavior)
    product_effects = {
        "BioProduct-A": {"yield_base": 85, "viability_base": 93, "impurity_base": 1.2},
        "BioProduct-B": {"yield_base": 82, "viability_base": 91, "impurity_base": 1.5},
        "CellTherapy-X": {"yield_base": 78, "viability_base": 89, "impurity_base": 2.0},
        "GeneVector-Z": {"yield_base": 80, "viability_base": 90, "impurity_base": 1.8}
    }

    # Site effects (Dublin has drift issue)
    site_effects = {
        "Site-Boston": {"yield_adj": 0, "temp_excursion_prob": 0.05},
        "Site-Dublin": {"yield_adj": -2, "temp_excursion_prob": 0.12},  # Slightly worse
        "Site-Singapore": {"yield_adj": 1, "temp_excursion_prob": 0.04}
    }

    # Step-specific adjustments
    step_effects = {
        "Thaw & Seed": {"variability": 1.2, "impurity_mult": 0.5},
        "Expansion Day 3": {"variability": 1.0, "impurity_mult": 0.6},
        "Expansion Day 7": {"variability": 1.0, "impurity_mult": 0.8},
        "Harvest": {"variability": 1.5, "impurity_mult": 1.2},
        "Formulation": {"variability": 1.1, "impurity_mult": 1.0},
        "Fill & Finish": {"variability": 1.3, "impurity_mult": 1.1},
        "QC Release": {"variability": 0.5, "impurity_mult": 1.0}
    }

    # Shift effects (night shift slightly more variable)
    shift_variability = {"Day": 1.0, "Evening": 1.05, "Night": 1.15}

    # Time-based drift for Dublin site (simulating equipment degradation)
    time_progress = batch_idx / n_batches
    dublin_drift = -3 * time_progress if site == "Site-Dublin" else 0

    # Bad lot events (every ~50 batches on average)
    bad_lot_event = np.random.random() < 0.02
    bad_lot_penalty = -8 if bad_lot_event else 0

    # Maintenance reset (improves performance temporarily)
    maintenance_event = np.random.random() < 0.01
    maintenance_bonus = 3 if maintenance_event else 0

    # Get base parameters
    p_eff = product_effects[product]
    s_eff = site_effects[site]
    st_eff = step_effects[process_step]
    sh_var = shift_variability[shift]

    # Calculate metrics with correlations
    base_yield = (
        p_eff["yield_base"] +
        s_eff["yield_adj"] +
        dublin_drift +
        bad_lot_penalty +
        maintenance_bonus
    )

    # Hold time affects yield (longer hold = lower yield)
    hold_time = max(0, np.random.exponential(4) + np.random.normal(0, 1))
    hold_time = min(hold_time, 30)  # Cap at 30 hours
    hold_time_penalty = -0.5 * max(0, hold_time - 8)  # Penalty for hold > 8 hrs

    # Generate correlated metrics
    noise_yield = np.random.normal(0, 3 * st_eff["variability"] * sh_var)
    yield_pct = np.clip(base_yield + hold_time_penalty + noise_yield, 40, 100)

    # Viability correlates with yield
    viability_base = p_eff["viability_base"] + 0.3 * (yield_pct - base_yield)
    viability_pct = np.clip(
        viability_base + np.random.normal(0, 2 * st_eff["variability"] * sh_var),
        50, 100
    )

    # Potency proxy
    potency_proxy = np.clip(
        100 + 0.2 * (yield_pct - 80) + np.random.normal(0, 5 * st_eff["variability"]),
        60, 140
    )

    # Impurity (inversely correlated with yield)
    impurity_base = p_eff["impurity_base"] * st_eff["impurity_mult"]
    impurity_pct = np.clip(
        impurity_base - 0.05 * (yield_pct - 80) + np.random.exponential(0.5),
        0.1, 10
    )

    # Cycle time
    cycle_time_base = 36 + step_idx * 2
    cycle_time_hrs = max(20, cycle_time_base + np.random.normal(0, 4 * sh_var))

    # Temperature excursions
    temp_excursion_count = np.random.poisson(s_eff["temp_excursion_prob"] * 2)
    if temp_excursion_count > 0:
        yield_pct -= temp_excursion_count * 2
        viability_pct -= temp_excursion_count * 1.5

    # Deviation count
    deviation_count = np.random.poisson(0.3 + 0.1 * (bad_lot_event + (shift == "Night")))

    # Pass/fail determination
    passed_qc = (
        yield_pct >= SPEC_LIMITS["yield_pct"]["LSL"] and
        viability_pct >= SPEC_LIMITS["viability_pct"]["LSL"] and
        impurity_pct <= SPEC_LIMITS["impurity_pct"]["USL"] and
        potency_proxy >= SPEC_LIMITS["potency_proxy"]["LSL"] and
        potency_proxy <= SPEC_LIMITS["potency_proxy"]["USL"]
    )

    return {
        "batch_id": batch_id,
        "lot_id": lot_id,
        "batch_date": batch_date,
        "product": product,
        "site": site,
        "line": line,
        "shift": shift,
        "operator_id": operator,
        "process_step": process_step,
        "step_sequence": step_idx + 1,
        "yield_pct": round(yield_pct, 2),
        "viability_pct": round(viability_pct, 2),
        "potency_proxy": round(potency_proxy, 2),
        "impurity_pct": round(impurity_pct, 3),
        "cycle_time_hrs": round(cycle_time_hrs, 1),
        "hold_time_hrs": round(hold_time, 1),
        "temp_excursion_count": temp_excursion_count,
        "deviation_count": deviation_count,
        "passed_qc": passed_qc,
        "bad_lot_event": bad_lot_event,
        "maintenance_event": maintenance_event
    }


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metrics and columns to the dataframe."""

    # Add week and month columns
    df["batch_week"] = df["batch_date"].dt.isocalendar().week
    df["batch_month"] = df["batch_date"].dt.month
    df["batch_quarter"] = df["batch_date"].dt.quarter
    df["day_of_week"] = df["batch_date"].dt.day_name()

    # Sort by date and step sequence
    df = df.sort_values(["batch_date", "batch_id", "step_sequence"])

    return df


def get_batch_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate step-level data to batch-level summary.
    Uses final step (QC Release) metrics for most values.
    """
    # Get final step metrics
    final_step = df[df["process_step"] == "QC Release"].copy()

    # Aggregate across all steps for counts
    agg_metrics = df.groupby("batch_id").agg({
        "temp_excursion_count": "sum",
        "deviation_count": "sum",
        "cycle_time_hrs": "sum",
        "hold_time_hrs": "sum"
    }).reset_index()
    agg_metrics.columns = [
        "batch_id", "total_temp_excursions", "total_deviations",
        "total_cycle_time_hrs", "total_hold_time_hrs"
    ]

    # Merge final step with aggregated metrics
    batch_summary = final_step.merge(agg_metrics, on="batch_id", how="left")

    # Select and rename columns
    batch_summary = batch_summary[[
        "batch_id", "lot_id", "batch_date", "product", "site", "line",
        "shift", "operator_id", "yield_pct", "viability_pct", "potency_proxy",
        "impurity_pct", "total_cycle_time_hrs", "total_hold_time_hrs",
        "total_temp_excursions", "total_deviations", "passed_qc",
        "bad_lot_event", "maintenance_event", "batch_week", "batch_month",
        "batch_quarter", "day_of_week"
    ]]

    return batch_summary


def get_specification_limits() -> Dict:
    """Return specification limits for metrics."""
    return SPEC_LIMITS.copy()


def get_products() -> List[str]:
    """Return list of products."""
    return PRODUCTS.copy()


def get_sites() -> List[str]:
    """Return list of sites."""
    return SITES.copy()


def get_process_steps() -> List[str]:
    """Return list of process steps."""
    return PROCESS_STEPS.copy()


def get_shifts() -> List[str]:
    """Return list of shifts."""
    return SHIFTS.copy()


# Cache for generated data
_cached_data: Optional[pd.DataFrame] = None
_cached_batch_summary: Optional[pd.DataFrame] = None


def get_cached_data(force_regenerate: bool = False) -> pd.DataFrame:
    """Get cached step-level data or generate if not available."""
    global _cached_data
    if _cached_data is None or force_regenerate:
        _cached_data = generate_batch_data()
    return _cached_data.copy()


def get_cached_batch_summary(force_regenerate: bool = False) -> pd.DataFrame:
    """Get cached batch summary or generate if not available."""
    global _cached_batch_summary
    if _cached_batch_summary is None or force_regenerate:
        data = get_cached_data(force_regenerate)
        _cached_batch_summary = get_batch_summary(data)
    return _cached_batch_summary.copy()
