"""
Visualization Components Module

Professional Plotly charts for manufacturing analytics dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any

from src.analytics.spc import (
    ControlChartResult, MRChartResult, EWMAResult, CUSUMResult,
    CapabilityResult, AlertType
)


# Color palette for consistent styling
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffbb00",
    "info": "#17becf",
    "gray": "#7f7f7f",
    "light_gray": "#d3d3d3",
    "centerline": "#2ca02c",
    "control_limit": "#d62728",
    "spec_limit": "#9467bd",
    "ewma": "#e377c2",
    "cusum_pos": "#ff7f0e",
    "cusum_neg": "#1f77b4"
}

LAYOUT_DEFAULTS = {
    "font": {"family": "Arial, sans-serif", "size": 12},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
    "hovermode": "x unified"
}


def create_control_chart(
    result: ControlChartResult,
    title: str = "Individuals Control Chart",
    x_labels: Optional[List[str]] = None,
    show_zones: bool = True,
    height: int = 400
) -> go.Figure:
    """
    Create an Individuals (I) control chart with control limits and alerts.

    Parameters:
    -----------
    result : ControlChartResult
        Control chart calculation results
    title : str
        Chart title
    x_labels : List[str], optional
        X-axis labels (e.g., dates or batch IDs)
    show_zones : bool
        Whether to show 1σ and 2σ zones
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    n = len(result.values)
    x = list(range(n)) if x_labels is None else x_labels

    fig = go.Figure()

    # Add zones if requested
    if show_zones and result.sigma > 0:
        # 1σ zone
        fig.add_hrect(
            y0=result.centerline - result.sigma,
            y1=result.centerline + result.sigma,
            fillcolor="rgba(0, 255, 0, 0.1)",
            line_width=0,
            layer="below"
        )
        # 2σ zone
        fig.add_hrect(
            y0=result.centerline + result.sigma,
            y1=result.centerline + 2 * result.sigma,
            fillcolor="rgba(255, 255, 0, 0.1)",
            line_width=0,
            layer="below"
        )
        fig.add_hrect(
            y0=result.centerline - 2 * result.sigma,
            y1=result.centerline - result.sigma,
            fillcolor="rgba(255, 255, 0, 0.1)",
            line_width=0,
            layer="below"
        )
        # 3σ zone
        fig.add_hrect(
            y0=result.centerline + 2 * result.sigma,
            y1=result.ucl,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            layer="below"
        )
        fig.add_hrect(
            y0=result.lcl,
            y1=result.centerline - 2 * result.sigma,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            layer="below"
        )

    # Data points
    alert_indices = set(a[0] for a in result.alerts)
    normal_x = [x[i] for i in range(n) if i not in alert_indices]
    normal_y = [result.values[i] for i in range(n) if i not in alert_indices]
    alert_x = [x[i] for i in range(n) if i in alert_indices]
    alert_y = [result.values[i] for i in range(n) if i in alert_indices]

    # Normal points
    fig.add_trace(go.Scatter(
        x=normal_x,
        y=normal_y,
        mode="markers+lines",
        name="Observations",
        marker={"color": COLORS["primary"], "size": 8},
        line={"color": COLORS["primary"], "width": 1}
    ))

    # Alert points
    if alert_x:
        fig.add_trace(go.Scatter(
            x=alert_x,
            y=alert_y,
            mode="markers",
            name="Alerts",
            marker={"color": COLORS["danger"], "size": 12, "symbol": "diamond"}
        ))

    # Control limits
    fig.add_hline(y=result.centerline, line_dash="solid",
                  line_color=COLORS["centerline"], line_width=2,
                  annotation_text=f"CL = {result.centerline:.2f}")
    fig.add_hline(y=result.ucl, line_dash="dash",
                  line_color=COLORS["control_limit"], line_width=2,
                  annotation_text=f"UCL = {result.ucl:.2f}")
    fig.add_hline(y=result.lcl, line_dash="dash",
                  line_color=COLORS["control_limit"], line_width=2,
                  annotation_text=f"LCL = {result.lcl:.2f}")

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Observation",
        yaxis_title="Value",
        height=height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        **LAYOUT_DEFAULTS
    )

    return fig


def create_mr_chart(
    result: MRChartResult,
    title: str = "Moving Range Chart",
    x_labels: Optional[List[str]] = None,
    height: int = 300
) -> go.Figure:
    """
    Create a Moving Range (MR) control chart.

    Parameters:
    -----------
    result : MRChartResult
        Moving range chart calculation results
    title : str
        Chart title
    x_labels : List[str], optional
        X-axis labels
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    n = len(result.mr_values)
    x = list(range(n)) if x_labels is None else x_labels[:n]

    fig = go.Figure()

    # MR values
    alert_indices = set(a[0] for a in result.alerts)
    normal_x = [x[i] for i in range(n) if i not in alert_indices]
    normal_y = [result.mr_values[i] for i in range(n) if i not in alert_indices]
    alert_x = [x[i] for i in range(n) if i in alert_indices]
    alert_y = [result.mr_values[i] for i in range(n) if i in alert_indices]

    fig.add_trace(go.Scatter(
        x=normal_x,
        y=normal_y,
        mode="markers+lines",
        name="Moving Range",
        marker={"color": COLORS["secondary"], "size": 6},
        line={"color": COLORS["secondary"], "width": 1}
    ))

    if alert_x:
        fig.add_trace(go.Scatter(
            x=alert_x,
            y=alert_y,
            mode="markers",
            name="Alerts",
            marker={"color": COLORS["danger"], "size": 10, "symbol": "diamond"}
        ))

    # Control limits
    fig.add_hline(y=result.centerline, line_dash="solid",
                  line_color=COLORS["centerline"], line_width=2,
                  annotation_text=f"MR̄ = {result.centerline:.2f}")
    fig.add_hline(y=result.ucl, line_dash="dash",
                  line_color=COLORS["control_limit"], line_width=2,
                  annotation_text=f"UCL = {result.ucl:.2f}")

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Observation",
        yaxis_title="Moving Range",
        height=height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        **LAYOUT_DEFAULTS
    )

    return fig


def create_imr_chart(
    i_result: ControlChartResult,
    mr_result: MRChartResult,
    title: str = "I-MR Control Chart",
    x_labels: Optional[List[str]] = None,
    height: int = 600
) -> go.Figure:
    """
    Create combined Individuals and Moving Range control chart.

    Parameters:
    -----------
    i_result : ControlChartResult
        Individuals chart calculation results
    mr_result : MRChartResult
        Moving range chart calculation results
    title : str
        Chart title
    x_labels : List[str], optional
        X-axis labels
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=("Individuals Chart", "Moving Range Chart"),
        vertical_spacing=0.12
    )

    n = len(i_result.values)
    x = list(range(n)) if x_labels is None else x_labels

    # --- Individuals Chart (Top) ---
    alert_indices = set(a[0] for a in i_result.alerts)
    normal_x = [x[i] for i in range(n) if i not in alert_indices]
    normal_y = [i_result.values[i] for i in range(n) if i not in alert_indices]
    alert_x = [x[i] for i in range(n) if i in alert_indices]
    alert_y = [i_result.values[i] for i in range(n) if i in alert_indices]

    fig.add_trace(go.Scatter(
        x=normal_x, y=normal_y,
        mode="markers+lines", name="Observations",
        marker={"color": COLORS["primary"], "size": 8},
        line={"color": COLORS["primary"], "width": 1}
    ), row=1, col=1)

    if alert_x:
        fig.add_trace(go.Scatter(
            x=alert_x, y=alert_y,
            mode="markers", name="Alerts",
            marker={"color": COLORS["danger"], "size": 12, "symbol": "diamond"}
        ), row=1, col=1)

    # I chart control limits
    for y, name, color, dash in [
        (i_result.centerline, "CL", COLORS["centerline"], "solid"),
        (i_result.ucl, "UCL", COLORS["control_limit"], "dash"),
        (i_result.lcl, "LCL", COLORS["control_limit"], "dash")
    ]:
        fig.add_hline(y=y, line_dash=dash, line_color=color, line_width=2, row=1, col=1)

    # --- Moving Range Chart (Bottom) ---
    n_mr = len(mr_result.mr_values)
    x_mr = x[1:n_mr+1] if x_labels is not None else list(range(1, n_mr + 1))

    mr_alert_indices = set(a[0] for a in mr_result.alerts)
    mr_normal_x = [x_mr[i] for i in range(n_mr) if i not in mr_alert_indices]
    mr_normal_y = [mr_result.mr_values[i] for i in range(n_mr) if i not in mr_alert_indices]
    mr_alert_x = [x_mr[i] for i in range(n_mr) if i in mr_alert_indices]
    mr_alert_y = [mr_result.mr_values[i] for i in range(n_mr) if i in mr_alert_indices]

    fig.add_trace(go.Scatter(
        x=mr_normal_x, y=mr_normal_y,
        mode="markers+lines", name="Moving Range",
        marker={"color": COLORS["secondary"], "size": 6},
        line={"color": COLORS["secondary"], "width": 1},
        showlegend=False
    ), row=2, col=1)

    if mr_alert_x:
        fig.add_trace(go.Scatter(
            x=mr_alert_x, y=mr_alert_y,
            mode="markers", name="MR Alerts",
            marker={"color": COLORS["danger"], "size": 10, "symbol": "diamond"},
            showlegend=False
        ), row=2, col=1)

    # MR chart control limits
    fig.add_hline(y=mr_result.centerline, line_dash="solid",
                  line_color=COLORS["centerline"], line_width=2, row=2, col=1)
    fig.add_hline(y=mr_result.ucl, line_dash="dash",
                  line_color=COLORS["control_limit"], line_width=2, row=2, col=1)

    fig.update_layout(
        title={"text": title, "x": 0.5},
        height=height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        **LAYOUT_DEFAULTS
    )

    fig.update_xaxes(title_text="Observation", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Moving Range", row=2, col=1)

    return fig


def create_ewma_chart(
    result: EWMAResult,
    title: str = "EWMA Control Chart",
    x_labels: Optional[List[str]] = None,
    height: int = 400
) -> go.Figure:
    """
    Create EWMA control chart.

    Parameters:
    -----------
    result : EWMAResult
        EWMA calculation results
    title : str
        Chart title
    x_labels : List[str], optional
        X-axis labels
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    n = len(result.ewma_values)
    x = list(range(n)) if x_labels is None else x_labels

    fig = go.Figure()

    # Control limit bands
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(result.ucl_values) + list(result.lcl_values)[::-1],
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.1)",
        line={"width": 0},
        name="Control Limits",
        showlegend=False
    ))

    # EWMA line
    alert_indices = set(a[0] for a in result.alerts)

    fig.add_trace(go.Scatter(
        x=x,
        y=result.ewma_values,
        mode="lines+markers",
        name=f"EWMA (λ={result.lambda_param})",
        line={"color": COLORS["ewma"], "width": 2},
        marker={"size": 6, "color": [COLORS["danger"] if i in alert_indices else COLORS["ewma"] for i in range(n)]}
    ))

    # UCL and LCL lines
    fig.add_trace(go.Scatter(
        x=x, y=result.ucl_values,
        mode="lines", name="UCL",
        line={"color": COLORS["control_limit"], "dash": "dash", "width": 1}
    ))
    fig.add_trace(go.Scatter(
        x=x, y=result.lcl_values,
        mode="lines", name="LCL",
        line={"color": COLORS["control_limit"], "dash": "dash", "width": 1}
    ))

    # Centerline
    fig.add_hline(y=result.centerline, line_dash="solid",
                  line_color=COLORS["centerline"], line_width=2,
                  annotation_text=f"Target = {result.centerline:.2f}")

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Observation",
        yaxis_title="EWMA Value",
        height=height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        **LAYOUT_DEFAULTS
    )

    return fig


def create_cusum_chart(
    result: CUSUMResult,
    title: str = "CUSUM Control Chart",
    x_labels: Optional[List[str]] = None,
    height: int = 400
) -> go.Figure:
    """
    Create CUSUM control chart.

    Parameters:
    -----------
    result : CUSUMResult
        CUSUM calculation results
    title : str
        Chart title
    x_labels : List[str], optional
        X-axis labels
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    n = len(result.cusum_pos)
    x = list(range(n)) if x_labels is None else x_labels

    fig = go.Figure()

    # Positive CUSUM
    fig.add_trace(go.Scatter(
        x=x, y=result.cusum_pos,
        mode="lines+markers", name="CUSUM+ (upward)",
        line={"color": COLORS["cusum_pos"], "width": 2},
        marker={"size": 5}
    ))

    # Negative CUSUM
    fig.add_trace(go.Scatter(
        x=x, y=result.cusum_neg,
        mode="lines+markers", name="CUSUM- (downward)",
        line={"color": COLORS["cusum_neg"], "width": 2},
        marker={"size": 5}
    ))

    # Decision boundary
    fig.add_hline(y=result.h_limit, line_dash="dash",
                  line_color=COLORS["control_limit"], line_width=2,
                  annotation_text=f"h = {result.h_limit}")
    fig.add_hline(y=0, line_dash="solid",
                  line_color=COLORS["gray"], line_width=1)

    # Mark alerts
    for idx, alert_type, direction in result.alerts:
        y_val = result.cusum_pos[idx] if direction == "positive" else result.cusum_neg[idx]
        fig.add_trace(go.Scatter(
            x=[x[idx]], y=[y_val],
            mode="markers", name=f"Alert ({direction})",
            marker={"color": COLORS["danger"], "size": 12, "symbol": "star"},
            showlegend=False
        ))

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Observation",
        yaxis_title="CUSUM Value",
        height=height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        **LAYOUT_DEFAULTS
    )

    return fig


def create_capability_histogram(
    data: np.ndarray,
    result: CapabilityResult,
    title: str = "Process Capability Analysis",
    height: int = 400
) -> go.Figure:
    """
    Create process capability histogram with specification limits.

    Parameters:
    -----------
    data : np.ndarray
        Array of measurements
    result : CapabilityResult
        Capability analysis results
    title : str
        Chart title
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name="Distribution",
        marker_color=COLORS["primary"],
        opacity=0.7
    ))

    # Normal distribution overlay
    x_range = np.linspace(data.min(), data.max(), 100)
    from scipy import stats
    pdf = stats.norm.pdf(x_range, result.mean, result.std)
    # Scale PDF to match histogram
    bin_width = (data.max() - data.min()) / 30
    pdf_scaled = pdf * len(data) * bin_width

    fig.add_trace(go.Scatter(
        x=x_range, y=pdf_scaled,
        mode="lines", name="Normal Fit",
        line={"color": COLORS["info"], "width": 2}
    ))

    # Specification limits
    if result.lsl is not None:
        fig.add_vline(x=result.lsl, line_dash="dash",
                      line_color=COLORS["spec_limit"], line_width=2,
                      annotation_text=f"LSL = {result.lsl}")
    if result.usl is not None:
        fig.add_vline(x=result.usl, line_dash="dash",
                      line_color=COLORS["spec_limit"], line_width=2,
                      annotation_text=f"USL = {result.usl}")
    if result.target is not None:
        fig.add_vline(x=result.target, line_dash="solid",
                      line_color=COLORS["success"], line_width=2,
                      annotation_text=f"Target = {result.target}")

    # Mean line
    fig.add_vline(x=result.mean, line_dash="dot",
                  line_color=COLORS["primary"], line_width=2,
                  annotation_text=f"Mean = {result.mean:.2f}")

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Value",
        yaxis_title="Count",
        height=height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        **LAYOUT_DEFAULTS
    )

    return fig


def create_run_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Run Chart",
    color_column: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """
    Create a simple run chart (time series plot).

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with data
    x_column : str
        Column for x-axis
    y_column : str
        Column for y-axis
    title : str
        Chart title
    color_column : str, optional
        Column for color coding
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.line(
        df, x=x_column, y=y_column,
        color=color_column,
        title=title,
        markers=True,
        height=height
    )

    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(title={"x": 0.5})

    return fig


def create_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str = "Time Series",
    show_trend: bool = True,
    height: int = 400
) -> go.Figure:
    """
    Create time series chart with optional trend line.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with time series data
    date_column : str
        Date column name
    value_column : str
        Value column name
    title : str
        Chart title
    show_trend : bool
        Whether to show trend line
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    from scipy import stats

    df_sorted = df.sort_values(date_column)

    fig = go.Figure()

    # Main time series
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=df_sorted[value_column],
        mode="markers+lines",
        name=value_column,
        marker={"size": 6, "color": COLORS["primary"]},
        line={"width": 1, "color": COLORS["primary"]}
    ))

    # Trend line
    if show_trend and len(df_sorted) >= 3:
        x_numeric = (pd.to_datetime(df_sorted[date_column]) -
                     pd.to_datetime(df_sorted[date_column]).min()).dt.days.values
        y = df_sorted[value_column].values

        slope, intercept, _, _, _ = stats.linregress(x_numeric, y)
        trend_y = intercept + slope * x_numeric

        fig.add_trace(go.Scatter(
            x=df_sorted[date_column],
            y=trend_y,
            mode="lines",
            name="Trend",
            line={"width": 2, "color": COLORS["danger"], "dash": "dash"}
        ))

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Date",
        yaxis_title=value_column,
        height=height,
        showlegend=True,
        **LAYOUT_DEFAULTS
    )

    return fig


def create_kpi_gauge(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    thresholds: Optional[Dict[str, float]] = None,
    suffix: str = "%"
) -> go.Figure:
    """
    Create a KPI gauge chart.

    Parameters:
    -----------
    value : float
        Current value
    title : str
        Gauge title
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    thresholds : Dict[str, float], optional
        Threshold values for color bands
    suffix : str
        Value suffix

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if thresholds is None:
        thresholds = {"red": 60, "yellow": 80, "green": 100}

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        number={"suffix": suffix},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [min_val, thresholds.get("red", 60)], "color": "rgba(214, 39, 40, 0.3)"},
                {"range": [thresholds.get("red", 60), thresholds.get("yellow", 80)], "color": "rgba(255, 187, 0, 0.3)"},
                {"range": [thresholds.get("yellow", 80), max_val], "color": "rgba(44, 160, 44, 0.3)"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin={"l": 20, "r": 20, "t": 50, "b": 20}
    )

    return fig


def create_heatmap(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    title: str = "Heatmap",
    colorscale: str = "RdYlGn",
    height: int = 400
) -> go.Figure:
    """
    Create a heatmap chart.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with pivot table data
    x_column : str
        Column for x-axis categories
    y_column : str
        Column for y-axis categories
    value_column : str
        Column for values
    title : str
        Chart title
    colorscale : str
        Plotly colorscale name
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    pivot = df.pivot_table(
        values=value_column,
        index=y_column,
        columns=x_column,
        aggfunc="mean"
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=colorscale,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate=f"{x_column}: %{{x}}<br>{y_column}: %{{y}}<br>{value_column}: %{{z:.2f}}<extra></extra>"
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5},
        height=height,
        **LAYOUT_DEFAULTS
    )

    return fig


def create_box_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Box Plot",
    height: int = 400
) -> go.Figure:
    """
    Create a box plot for comparing distributions across groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with data
    x_column : str
        Column for grouping (x-axis)
    y_column : str
        Column for values (y-axis)
    title : str
        Chart title
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.box(
        df, x=x_column, y=y_column,
        title=title,
        color=x_column,
        height=height
    )

    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(
        title={"x": 0.5},
        showlegend=False
    )

    return fig


def create_scatter_matrix(
    df: pd.DataFrame,
    columns: List[str],
    color_column: Optional[str] = None,
    title: str = "Scatter Matrix",
    height: int = 600
) -> go.Figure:
    """
    Create a scatter plot matrix for exploring correlations.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with data
    columns : List[str]
        Columns to include in the matrix
    color_column : str, optional
        Column for color coding
    title : str
        Chart title
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color_column,
        title=title,
        height=height
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(title={"x": 0.5})

    return fig


def create_driver_importance_chart(
    drivers: List[Any],
    title: str = "Driver Importance",
    height: int = 400
) -> go.Figure:
    """
    Create horizontal bar chart showing driver importance.

    Parameters:
    -----------
    drivers : List[DriverResult]
        List of driver analysis results
    title : str
        Chart title
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if not drivers:
        fig = go.Figure()
        fig.add_annotation(text="No drivers to display", showarrow=False)
        return fig

    features = [d.feature.replace("_", " ").title() for d in drivers]
    importances = [d.importance for d in drivers]
    directions = [d.direction for d in drivers]
    colors = [COLORS["success"] if d == "positive" else COLORS["danger"] for d in directions]

    fig = go.Figure(go.Bar(
        y=features,
        x=importances,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in importances],
        textposition="outside"
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Importance Score",
        yaxis_title="",
        height=height,
        yaxis={"categoryorder": "total ascending"},
        **LAYOUT_DEFAULTS
    )

    return fig
