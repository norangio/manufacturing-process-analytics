"""
Driver Analysis Module

Implements correlation analysis and lightweight modeling
for identifying process drivers and trends.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    feature: str
    correlation: float
    p_value: float
    interpretation: str


@dataclass
class DriverResult:
    """Result of driver analysis."""
    feature: str
    importance: float
    direction: str
    interpretation: str


@dataclass
class TrendResult:
    """Result of trend analysis."""
    metric: str
    slope: float
    trend_direction: str
    r_squared: float
    interpretation: str


def calculate_correlations(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str]
) -> List[CorrelationResult]:
    """
    Calculate correlations between features and a target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame containing all variables
    target_column : str
        Name of the target variable column
    feature_columns : List[str]
        List of feature column names to correlate with target

    Returns:
    --------
    List[CorrelationResult]
        List of correlation results sorted by absolute correlation
    """
    from scipy import stats

    results = []

    target = df[target_column].dropna()

    for feature in feature_columns:
        if feature == target_column:
            continue

        # Get paired data
        mask = df[target_column].notna() & df[feature].notna()
        if mask.sum() < 3:
            continue

        x = df.loc[mask, feature].values
        y = df.loc[mask, target_column].values

        # Skip if no variation
        if np.std(x) == 0 or np.std(y) == 0:
            continue

        # Calculate Pearson correlation
        corr, p_value = stats.pearsonr(x, y)

        # Generate interpretation
        interpretation = interpret_correlation(feature, target_column, corr)

        results.append(CorrelationResult(
            feature=feature,
            correlation=corr,
            p_value=p_value,
            interpretation=interpretation
        ))

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x.correlation), reverse=True)

    return results


def interpret_correlation(feature: str, target: str, corr: float) -> str:
    """Generate plain-language interpretation of correlation."""
    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
    direction = "positive" if corr > 0 else "negative"

    # Feature name cleanup for readability
    feature_clean = feature.replace("_", " ").replace("pct", "%").replace("hrs", "hours")
    target_clean = target.replace("_", " ").replace("pct", "%")

    if direction == "positive":
        return f"{strength.capitalize()} positive relationship: as {feature_clean} increases, {target_clean} tends to increase."
    else:
        return f"{strength.capitalize()} negative relationship: as {feature_clean} increases, {target_clean} tends to decrease."


def analyze_drivers_regression(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    model_type: str = "linear"
) -> List[DriverResult]:
    """
    Analyze drivers using regression model.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame containing all variables
    target_column : str
        Name of the target variable column
    feature_columns : List[str]
        List of feature column names
    model_type : str
        Type of model: "linear" or "forest"

    Returns:
    --------
    List[DriverResult]
        List of driver results sorted by importance
    """
    # Prepare data
    valid_features = [f for f in feature_columns if f != target_column and f in df.columns]

    # Drop rows with any missing values
    subset = df[[target_column] + valid_features].dropna()

    if len(subset) < 10:
        return []

    X = subset[valid_features].values
    y = subset[target_column].values

    # Standardize features for comparable coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if model_type == "forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled, y)
            importances = model.feature_importances_

            # Determine direction from correlations
            directions = []
            for i, feat in enumerate(valid_features):
                corr = np.corrcoef(X_scaled[:, i], y)[0, 1]
                directions.append("positive" if corr > 0 else "negative")

        else:  # linear
            model = LinearRegression()
            model.fit(X_scaled, y)
            importances = np.abs(model.coef_)
            directions = ["positive" if c > 0 else "negative" for c in model.coef_]

    # Create results
    results = []
    for i, feature in enumerate(valid_features):
        importance = importances[i]
        direction = directions[i]
        interpretation = interpret_driver(feature, target_column, importance, direction)

        results.append(DriverResult(
            feature=feature,
            importance=importance,
            direction=direction,
            interpretation=interpretation
        ))

    # Sort by importance
    results.sort(key=lambda x: x.importance, reverse=True)

    return results


def interpret_driver(feature: str, target: str, importance: float, direction: str) -> str:
    """Generate plain-language interpretation of driver importance."""
    feature_clean = feature.replace("_", " ").replace("pct", "%").replace("hrs", "hours")
    target_clean = target.replace("_", " ").replace("pct", "%")

    strength = "major" if importance > 0.3 else "moderate" if importance > 0.1 else "minor"

    if direction == "positive":
        return f"{feature_clean.capitalize()} is a {strength} driver: higher values are associated with higher {target_clean}."
    else:
        return f"{feature_clean.capitalize()} is a {strength} driver: higher values are associated with lower {target_clean}."


def analyze_time_trends(
    df: pd.DataFrame,
    date_column: str,
    metric_columns: List[str],
    group_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze time-based trends for metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with time series data
    date_column : str
        Name of the date column
    metric_columns : List[str]
        List of metric columns to analyze
    group_column : str, optional
        Column to group by (e.g., site, product)

    Returns:
    --------
    pd.DataFrame
        Trend analysis results
    """
    from scipy import stats

    # Convert dates to numeric for regression
    df = df.copy()
    df['_date_num'] = (pd.to_datetime(df[date_column]) - pd.to_datetime(df[date_column]).min()).dt.days

    results = []

    groups = [None] if group_column is None else df[group_column].unique()

    for group in groups:
        if group is not None:
            subset = df[df[group_column] == group]
        else:
            subset = df

        for metric in metric_columns:
            if metric not in subset.columns:
                continue

            valid = subset[[metric, '_date_num']].dropna()
            if len(valid) < 5:
                continue

            x = valid['_date_num'].values
            y = valid[metric].values

            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Determine trend direction
            if p_value < 0.05:
                if slope > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
            else:
                trend = "stable"

            # Calculate percent change over period
            y_start = intercept
            y_end = intercept + slope * x.max()
            pct_change = ((y_end - y_start) / abs(y_start) * 100) if y_start != 0 else 0

            results.append({
                "group": group if group is not None else "All",
                "metric": metric,
                "slope_per_day": slope,
                "trend": trend,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "pct_change_over_period": pct_change,
                "interpretation": interpret_trend(metric, trend, pct_change, group)
            })

    return pd.DataFrame(results)


def interpret_trend(metric: str, trend: str, pct_change: float, group: Optional[str] = None) -> str:
    """Generate plain-language interpretation of trend."""
    metric_clean = metric.replace("_", " ").replace("pct", "%").replace("hrs", "hours")
    group_str = f" for {group}" if group and group != "All" else ""

    if trend == "stable":
        return f"{metric_clean.capitalize()}{group_str} has remained stable over the analysis period."
    elif trend == "increasing":
        return f"{metric_clean.capitalize()}{group_str} shows an upward trend, changing approximately {abs(pct_change):.1f}% over the period."
    else:
        return f"{metric_clean.capitalize()}{group_str} shows a downward trend, changing approximately {abs(pct_change):.1f}% over the period."


def detect_drift(
    df: pd.DataFrame,
    date_column: str,
    metric_column: str,
    window_size: int = 30,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect process drift over time.

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data
    date_column : str
        Date column name
    metric_column : str
        Metric to analyze for drift
    window_size : int
        Size of rolling window in days
    threshold : float
        Number of standard deviations for drift detection

    Returns:
    --------
    pd.DataFrame
        Drift detection results with rolling statistics
    """
    df = df.copy()
    df = df.sort_values(date_column)

    # Calculate overall baseline statistics
    baseline_mean = df[metric_column].mean()
    baseline_std = df[metric_column].std()

    # Calculate rolling statistics
    df['rolling_mean'] = df[metric_column].rolling(window=window_size, min_periods=5).mean()
    df['rolling_std'] = df[metric_column].rolling(window=window_size, min_periods=5).std()

    # Detect drift
    df['z_score'] = (df['rolling_mean'] - baseline_mean) / (baseline_std / np.sqrt(window_size))
    df['drift_detected'] = np.abs(df['z_score']) > threshold
    df['drift_direction'] = np.where(
        df['drift_detected'],
        np.where(df['z_score'] > 0, 'upward', 'downward'),
        'none'
    )

    return df[[date_column, metric_column, 'rolling_mean', 'rolling_std', 'z_score', 'drift_detected', 'drift_direction']]


def compare_groups(
    df: pd.DataFrame,
    metric_column: str,
    group_column: str
) -> pd.DataFrame:
    """
    Compare metric performance across groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with metric and group columns
    metric_column : str
        Metric to compare
    group_column : str
        Grouping variable

    Returns:
    --------
    pd.DataFrame
        Comparison statistics by group
    """
    from scipy import stats

    results = []

    groups = df[group_column].unique()
    overall_mean = df[metric_column].mean()
    overall_std = df[metric_column].std()

    for group in groups:
        subset = df[df[group_column] == group][metric_column].dropna()

        if len(subset) < 2:
            continue

        # T-test against overall mean
        t_stat, p_value = stats.ttest_1samp(subset, overall_mean)

        # Effect size (Cohen's d)
        cohens_d = (subset.mean() - overall_mean) / overall_std if overall_std > 0 else 0

        results.append({
            "group": group,
            "n": len(subset),
            "mean": subset.mean(),
            "std": subset.std(),
            "min": subset.min(),
            "max": subset.max(),
            "diff_from_overall": subset.mean() - overall_mean,
            "pct_diff": ((subset.mean() - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05
        })

    return pd.DataFrame(results).sort_values("mean", ascending=False)
