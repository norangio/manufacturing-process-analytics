"""
Statistical Process Control (SPC) Analytics Module

Implements control charts, detection rules, and capability indices
suitable for low-N manufacturing environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AlertType(Enum):
    """Types of SPC alerts/violations."""
    BEYOND_3_SIGMA = "Beyond 3σ"
    TWO_OF_THREE_BEYOND_2_SIGMA = "2 of 3 beyond 2σ"
    FOUR_OF_FIVE_BEYOND_1_SIGMA = "4 of 5 beyond 1σ"
    RUN_OF_8 = "8 points same side"
    TREND_OF_6 = "6 points trending"
    EWMA_SIGNAL = "EWMA signal"
    CUSUM_SIGNAL = "CUSUM signal"


@dataclass
class ControlChartResult:
    """Results from control chart analysis."""
    values: np.ndarray
    centerline: float
    ucl: float
    lcl: float
    alerts: List[Tuple[int, AlertType]]
    sigma: float


@dataclass
class MRChartResult:
    """Results from moving range chart."""
    mr_values: np.ndarray
    centerline: float
    ucl: float
    lcl: float
    alerts: List[Tuple[int, AlertType]]


@dataclass
class EWMAResult:
    """Results from EWMA analysis."""
    ewma_values: np.ndarray
    centerline: float
    ucl_values: np.ndarray
    lcl_values: np.ndarray
    alerts: List[Tuple[int, AlertType]]
    lambda_param: float


@dataclass
class CUSUMResult:
    """Results from CUSUM analysis."""
    cusum_pos: np.ndarray
    cusum_neg: np.ndarray
    h_limit: float
    alerts: List[Tuple[int, AlertType, str]]  # index, type, direction


@dataclass
class CapabilityResult:
    """Process capability analysis results."""
    ppk: float
    ppl: float
    ppu: float
    pp: float
    cpk: Optional[float]
    cp: Optional[float]
    mean: float
    std: float
    lsl: Optional[float]
    usl: Optional[float]
    target: Optional[float]
    within_spec_pct: float
    n_samples: int


def calculate_individuals_chart(
    data: np.ndarray,
    sigma_multiplier: float = 3.0
) -> ControlChartResult:
    """
    Calculate Individuals (I) control chart.

    For low-N manufacturing, uses moving range to estimate process variation.

    Parameters:
    -----------
    data : np.ndarray
        Array of individual measurements
    sigma_multiplier : float
        Multiplier for control limits (default 3.0)

    Returns:
    --------
    ControlChartResult
        Control chart statistics and alerts
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return ControlChartResult(
            values=data,
            centerline=np.mean(data) if n > 0 else 0,
            ucl=np.inf,
            lcl=-np.inf,
            alerts=[],
            sigma=0
        )

    # Calculate moving ranges
    mr = np.abs(np.diff(data))
    mr_bar = np.mean(mr)

    # d2 constant for n=2 (individuals chart uses consecutive pairs)
    d2 = 1.128

    # Estimate sigma from moving range
    sigma_mr = mr_bar / d2

    # Calculate control limits
    centerline = np.mean(data)
    ucl = centerline + sigma_multiplier * sigma_mr
    lcl = centerline - sigma_multiplier * sigma_mr

    # Detect alerts using Western Electric rules
    alerts = detect_western_electric_rules(data, centerline, sigma_mr)

    return ControlChartResult(
        values=data,
        centerline=centerline,
        ucl=ucl,
        lcl=lcl,
        alerts=alerts,
        sigma=sigma_mr
    )


def calculate_moving_range_chart(
    data: np.ndarray,
    sigma_multiplier: float = 3.0
) -> MRChartResult:
    """
    Calculate Moving Range (MR) control chart.

    Parameters:
    -----------
    data : np.ndarray
        Array of individual measurements
    sigma_multiplier : float
        Multiplier for control limits (default 3.0)

    Returns:
    --------
    MRChartResult
        Moving range chart statistics and alerts
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return MRChartResult(
            mr_values=np.array([]),
            centerline=0,
            ucl=np.inf,
            lcl=0,
            alerts=[]
        )

    # Calculate moving ranges
    mr = np.abs(np.diff(data))
    mr_bar = np.mean(mr)

    # D4 constant for n=2
    D4 = 3.267
    # D3 constant for n=2 (D3 = 0 for n < 7)
    D3 = 0

    ucl = D4 * mr_bar
    lcl = D3 * mr_bar

    # Detect points beyond UCL
    alerts = []
    for i, val in enumerate(mr):
        if val > ucl:
            alerts.append((i, AlertType.BEYOND_3_SIGMA))

    return MRChartResult(
        mr_values=mr,
        centerline=mr_bar,
        ucl=ucl,
        lcl=lcl,
        alerts=alerts
    )


def detect_western_electric_rules(
    data: np.ndarray,
    centerline: float,
    sigma: float
) -> List[Tuple[int, AlertType]]:
    """
    Apply Western Electric run rules for control chart alerts.

    Rules implemented:
    1. Any point beyond 3σ
    2. 2 of 3 consecutive points beyond 2σ (same side)
    3. 4 of 5 consecutive points beyond 1σ (same side)
    4. 8 consecutive points on same side of centerline
    5. 6 consecutive points trending up or down

    Parameters:
    -----------
    data : np.ndarray
        Array of measurements
    centerline : float
        Center line value
    sigma : float
        Process standard deviation estimate

    Returns:
    --------
    List[Tuple[int, AlertType]]
        List of (index, alert_type) tuples
    """
    alerts = []
    n = len(data)

    if n == 0 or sigma == 0:
        return alerts

    # Standardize data
    z_scores = (data - centerline) / sigma

    # Rule 1: Beyond 3σ
    for i, z in enumerate(z_scores):
        if abs(z) > 3:
            alerts.append((i, AlertType.BEYOND_3_SIGMA))

    # Rule 2: 2 of 3 beyond 2σ (same side)
    for i in range(2, n):
        window = z_scores[i-2:i+1]
        # Check positive side
        if sum(z > 2 for z in window) >= 2:
            alerts.append((i, AlertType.TWO_OF_THREE_BEYOND_2_SIGMA))
        # Check negative side
        elif sum(z < -2 for z in window) >= 2:
            alerts.append((i, AlertType.TWO_OF_THREE_BEYOND_2_SIGMA))

    # Rule 3: 4 of 5 beyond 1σ (same side)
    for i in range(4, n):
        window = z_scores[i-4:i+1]
        if sum(z > 1 for z in window) >= 4:
            alerts.append((i, AlertType.FOUR_OF_FIVE_BEYOND_1_SIGMA))
        elif sum(z < -1 for z in window) >= 4:
            alerts.append((i, AlertType.FOUR_OF_FIVE_BEYOND_1_SIGMA))

    # Rule 4: 8 consecutive points same side
    for i in range(7, n):
        window = z_scores[i-7:i+1]
        if all(z > 0 for z in window) or all(z < 0 for z in window):
            alerts.append((i, AlertType.RUN_OF_8))

    # Rule 5: 6 consecutive trending
    for i in range(5, n):
        window = data[i-5:i+1]
        diffs = np.diff(window)
        if all(d > 0 for d in diffs):  # All increasing
            alerts.append((i, AlertType.TREND_OF_6))
        elif all(d < 0 for d in diffs):  # All decreasing
            alerts.append((i, AlertType.TREND_OF_6))

    # Remove duplicates while preserving order
    seen = set()
    unique_alerts = []
    for alert in alerts:
        if alert not in seen:
            seen.add(alert)
            unique_alerts.append(alert)

    return unique_alerts


def calculate_ewma(
    data: np.ndarray,
    lambda_param: float = 0.2,
    L: float = 3.0,
    target: Optional[float] = None
) -> EWMAResult:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) control chart.

    EWMA is particularly effective for detecting small shifts in the process mean.

    Parameters:
    -----------
    data : np.ndarray
        Array of individual measurements
    lambda_param : float
        Smoothing parameter (0 < λ ≤ 1), default 0.2
    L : float
        Width of control limits in sigma units, default 3.0
    target : float, optional
        Target value (default uses data mean)

    Returns:
    --------
    EWMAResult
        EWMA chart statistics and alerts
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return EWMAResult(
            ewma_values=data,
            centerline=np.mean(data) if n > 0 else 0,
            ucl_values=np.full(n, np.inf),
            lcl_values=np.full(n, -np.inf),
            alerts=[],
            lambda_param=lambda_param
        )

    # Calculate process parameters
    mu = target if target is not None else np.mean(data)

    # Estimate sigma from moving range
    mr = np.abs(np.diff(data))
    sigma = np.mean(mr) / 1.128

    # Calculate EWMA values
    ewma_values = np.zeros(n)
    ewma_values[0] = lambda_param * data[0] + (1 - lambda_param) * mu

    for i in range(1, n):
        ewma_values[i] = lambda_param * data[i] + (1 - lambda_param) * ewma_values[i-1]

    # Calculate time-varying control limits
    # σ_EWMA = σ * sqrt(λ/(2-λ) * (1-(1-λ)^2i))
    i_values = np.arange(1, n + 1)
    sigma_ewma = sigma * np.sqrt(
        (lambda_param / (2 - lambda_param)) * (1 - (1 - lambda_param) ** (2 * i_values))
    )

    ucl_values = mu + L * sigma_ewma
    lcl_values = mu - L * sigma_ewma

    # Detect alerts
    alerts = []
    for i in range(n):
        if ewma_values[i] > ucl_values[i] or ewma_values[i] < lcl_values[i]:
            alerts.append((i, AlertType.EWMA_SIGNAL))

    return EWMAResult(
        ewma_values=ewma_values,
        centerline=mu,
        ucl_values=ucl_values,
        lcl_values=lcl_values,
        alerts=alerts,
        lambda_param=lambda_param
    )


def calculate_cusum(
    data: np.ndarray,
    k: float = 0.5,
    h: float = 5.0,
    target: Optional[float] = None
) -> CUSUMResult:
    """
    Calculate Cumulative Sum (CUSUM) control chart.

    CUSUM is effective for detecting sustained shifts from target.

    Parameters:
    -----------
    data : np.ndarray
        Array of individual measurements
    k : float
        Allowance value (slack), typically 0.5 sigma, default 0.5
    h : float
        Decision interval, typically 4-5 sigma, default 5.0
    target : float, optional
        Target value (default uses data mean)

    Returns:
    --------
    CUSUMResult
        CUSUM chart statistics and alerts
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return CUSUMResult(
            cusum_pos=np.zeros(n),
            cusum_neg=np.zeros(n),
            h_limit=h,
            alerts=[]
        )

    # Calculate process parameters
    mu = target if target is not None else np.mean(data)

    # Estimate sigma from moving range
    mr = np.abs(np.diff(data))
    sigma = np.mean(mr) / 1.128

    # Standardize data
    z = (data - mu) / sigma

    # Calculate CUSUM
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    for i in range(n):
        if i == 0:
            cusum_pos[i] = max(0, z[i] - k)
            cusum_neg[i] = max(0, -z[i] - k)
        else:
            cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - z[i] - k)

    # Detect alerts
    alerts = []
    for i in range(n):
        if cusum_pos[i] > h:
            alerts.append((i, AlertType.CUSUM_SIGNAL, "positive"))
        if cusum_neg[i] > h:
            alerts.append((i, AlertType.CUSUM_SIGNAL, "negative"))

    return CUSUMResult(
        cusum_pos=cusum_pos,
        cusum_neg=cusum_neg,
        h_limit=h,
        alerts=alerts
    )


def calculate_capability(
    data: np.ndarray,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    target: Optional[float] = None
) -> CapabilityResult:
    """
    Calculate process capability indices.

    Emphasizes Ppk (performance index) over Cpk for low-N manufacturing.

    Parameters:
    -----------
    data : np.ndarray
        Array of measurements
    lsl : float, optional
        Lower specification limit
    usl : float, optional
        Upper specification limit
    target : float, optional
        Target value

    Returns:
    --------
    CapabilityResult
        Capability analysis results

    Notes:
    ------
    Ppk uses overall standard deviation (σ total), appropriate for
    process performance assessment.
    Cpk uses within-subgroup variation, which requires rational subgrouping
    and may not be appropriate for all manufacturing contexts.
    """
    data = np.asarray(data)
    n = len(data)

    if n == 0:
        return CapabilityResult(
            ppk=0, ppl=0, ppu=0, pp=0,
            cpk=None, cp=None,
            mean=0, std=0,
            lsl=lsl, usl=usl, target=target,
            within_spec_pct=0, n_samples=0
        )

    mean = np.mean(data)
    std_overall = np.std(data, ddof=1)  # Sample std dev for Pp/Ppk

    # Estimate within-subgroup std from moving range (for Cp/Cpk)
    if n >= 2:
        mr = np.abs(np.diff(data))
        std_within = np.mean(mr) / 1.128
    else:
        std_within = std_overall

    # Calculate Pp and Ppk (using overall variation)
    ppl = ppu = pp = ppk = 0

    if std_overall > 0:
        if lsl is not None:
            ppl = (mean - lsl) / (3 * std_overall)
        else:
            ppl = np.inf

        if usl is not None:
            ppu = (usl - mean) / (3 * std_overall)
        else:
            ppu = np.inf

        ppk = min(ppl, ppu) if not np.isinf(min(ppl, ppu)) else 0

        if lsl is not None and usl is not None:
            pp = (usl - lsl) / (6 * std_overall)
        else:
            pp = 0

    # Calculate Cp and Cpk (using within-subgroup variation)
    cpl = cpu = cp = cpk = None

    if std_within > 0 and n >= 10:  # Only calculate with sufficient data
        if lsl is not None:
            cpl = (mean - lsl) / (3 * std_within)
        else:
            cpl = np.inf

        if usl is not None:
            cpu = (usl - mean) / (3 * std_within)
        else:
            cpu = np.inf

        if cpl is not None and cpu is not None:
            cpk = min(cpl, cpu) if not np.isinf(min(cpl, cpu)) else None

        if lsl is not None and usl is not None:
            cp = (usl - lsl) / (6 * std_within)

    # Calculate % within spec
    within_spec = 0
    if n > 0:
        in_spec = np.ones(n, dtype=bool)
        if lsl is not None:
            in_spec &= (data >= lsl)
        if usl is not None:
            in_spec &= (data <= usl)
        within_spec = np.sum(in_spec) / n * 100

    return CapabilityResult(
        ppk=ppk,
        ppl=ppl if not np.isinf(ppl) else None,
        ppu=ppu if not np.isinf(ppu) else None,
        pp=pp,
        cpk=cpk,
        cp=cp,
        mean=mean,
        std=std_overall,
        lsl=lsl,
        usl=usl,
        target=target,
        within_spec_pct=within_spec,
        n_samples=n
    )


def calculate_tolerance_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    coverage: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate a tolerance interval for the data.

    Parameters:
    -----------
    data : np.ndarray
        Array of measurements
    confidence : float
        Confidence level (e.g., 0.95)
    coverage : float
        Proportion of population to be covered (e.g., 0.95)

    Returns:
    --------
    Tuple[float, float]
        Lower and upper tolerance bounds
    """
    from scipy import stats

    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return (np.min(data), np.max(data)) if n > 0 else (0, 0)

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Use normal distribution tolerance factor
    # k = z_gamma * sqrt((n-1) / chi2_alpha)
    z_gamma = stats.norm.ppf((1 + coverage) / 2)
    chi2_alpha = stats.chi2.ppf(1 - confidence, n - 1)

    k = z_gamma * np.sqrt((n - 1) * (1 + 1/n) / chi2_alpha)

    lower = mean - k * std
    upper = mean + k * std

    return (lower, upper)


def get_alert_summary(
    df: pd.DataFrame,
    metric_column: str,
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate summary of SPC alerts for a metric across groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with metric values
    metric_column : str
        Column name of the metric to analyze
    group_columns : List[str], optional
        Columns to group by

    Returns:
    --------
    pd.DataFrame
        Summary of alerts by group
    """
    if group_columns is None:
        group_columns = []

    results = []

    if len(group_columns) == 0:
        # Analyze entire dataset
        values = df[metric_column].dropna().values
        chart = calculate_individuals_chart(values)
        ewma = calculate_ewma(values)

        results.append({
            "group": "All",
            "n_points": len(values),
            "n_alerts": len(chart.alerts),
            "n_ewma_signals": len(ewma.alerts),
            "mean": chart.centerline,
            "sigma": chart.sigma,
            "alert_types": [a[1].value for a in chart.alerts]
        })
    else:
        # Analyze by group
        for name, group in df.groupby(group_columns):
            values = group[metric_column].dropna().values
            if len(values) < 3:
                continue

            chart = calculate_individuals_chart(values)
            ewma = calculate_ewma(values)

            group_name = name if isinstance(name, str) else " / ".join(str(n) for n in name)

            results.append({
                "group": group_name,
                "n_points": len(values),
                "n_alerts": len(chart.alerts),
                "n_ewma_signals": len(ewma.alerts),
                "mean": chart.centerline,
                "sigma": chart.sigma,
                "alert_types": [a[1].value for a in chart.alerts]
            })

    return pd.DataFrame(results)
