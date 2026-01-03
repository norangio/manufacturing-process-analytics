"""
Manufacturing Analytics & SPC Dashboard

A professional demo application showcasing process monitoring,
statistical process control, and capability analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import local modules
from src.data.generator import (
    generate_batch_data, get_batch_summary, get_specification_limits,
    get_products, get_sites, get_process_steps, get_shifts, RANDOM_SEED
)
from src.analytics.spc import (
    calculate_individuals_chart, calculate_moving_range_chart,
    calculate_ewma, calculate_cusum, calculate_capability,
    calculate_tolerance_interval, get_alert_summary, AlertType
)
from src.analytics.drivers import (
    calculate_correlations, analyze_drivers_regression,
    analyze_time_trends, compare_groups
)
from src.viz.charts import (
    create_control_chart, create_mr_chart, create_imr_chart,
    create_ewma_chart, create_cusum_chart, create_capability_histogram,
    create_run_chart, create_time_series, create_kpi_gauge,
    create_heatmap, create_box_plot, create_driver_importance_chart
)

# Page configuration
st.set_page_config(
    page_title="Manufacturing Analytics Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #ffc107;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    """Load and cache synthetic manufacturing data."""
    step_data = generate_batch_data(n_batches=500, seed=RANDOM_SEED)
    batch_data = get_batch_summary(step_data)
    return step_data, batch_data


def apply_filters(df, products, sites, date_range, shifts):
    """Apply global filters to dataframe."""
    filtered = df.copy()
    if products:
        filtered = filtered[filtered["product"].isin(products)]
    if sites:
        filtered = filtered[filtered["site"].isin(sites)]
    if shifts:
        filtered = filtered[filtered["shift"].isin(shifts)]
    if date_range:
        filtered = filtered[
            (filtered["batch_date"] >= pd.to_datetime(date_range[0])) &
            (filtered["batch_date"] <= pd.to_datetime(date_range[1]))
        ]
    return filtered


def render_sidebar():
    """Render sidebar with navigation and filters."""
    with st.sidebar:
        st.markdown("## 🏭 Manufacturing Analytics")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigate to:",
            ["📊 Overview", "🔍 Batch Explorer", "📈 SPC / Control Charts",
             "🎯 Capability Analysis", "📉 Trends & Drivers", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Global Filters")

        # Get filter options
        products = st.multiselect("Product", get_products(), default=[])
        sites = st.multiselect("Site", get_sites(), default=[])
        shifts = st.multiselect("Shift", get_shifts(), default=[])

        # Date range
        date_range = st.date_input(
            "Date Range",
            value=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2024, 12, 31)
        )

        st.markdown("---")
        st.caption(f"Data generated with seed: {RANDOM_SEED}")
        st.caption("This is a demo with synthetic data")

    return page, products, sites, date_range, shifts


def render_overview(batch_df, step_df):
    """Render the Overview page."""
    st.markdown('<p class="main-header">Manufacturing Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Key performance indicators and alerts summary</p>', unsafe_allow_html=True)

    # KPI metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_batches = len(batch_df)
    unique_lots = batch_df["lot_id"].nunique()
    avg_yield = batch_df["yield_pct"].mean()
    pass_rate = (batch_df["passed_qc"].sum() / total_batches * 100) if total_batches > 0 else 0
    total_deviations = batch_df["total_deviations"].sum()

    with col1:
        st.metric("Total Batches", f"{total_batches:,}")
    with col2:
        st.metric("Unique Lots", f"{unique_lots}")
    with col3:
        st.metric("Avg Yield", f"{avg_yield:.1f}%")
    with col4:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col5:
        st.metric("Total Deviations", f"{total_deviations:,}")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Yield Distribution by Product")
        fig = create_box_plot(batch_df, "product", "yield_pct", "Yield % by Product")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Pass Rate by Site")
        site_stats = batch_df.groupby("site").agg(
            pass_rate=("passed_qc", lambda x: x.sum()/len(x)*100),
            count=("batch_id", "count")
        ).reset_index()
        import plotly.express as px
        fig = px.bar(site_stats, x="site", y="pass_rate", color="site",
                     title="QC Pass Rate by Site (%)", text_auto=".1f")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Alerts section
    st.markdown("---")
    st.subheader("🚨 Recent Alerts Summary")

    # Get SPC alerts for yield
    alert_summary = get_alert_summary(batch_df, "yield_pct", ["site"])

    if not alert_summary.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                alert_summary[["group", "n_points", "n_alerts", "n_ewma_signals", "mean"]].rename(
                    columns={"group": "Site", "n_points": "Batches", "n_alerts": "Control Alerts",
                             "n_ewma_signals": "EWMA Signals", "mean": "Mean Yield"}
                ),
                hide_index=True,
                use_container_width=True
            )
        with col2:
            total_alerts = alert_summary["n_alerts"].sum()
            st.metric("Total Control Chart Alerts", total_alerts)
            if total_alerts > 10:
                st.warning("High alert count - investigate process stability")

    # Trend over time
    st.markdown("---")
    st.subheader("Yield Trend Over Time")
    daily_yield = batch_df.groupby("batch_date")["yield_pct"].mean().reset_index()
    fig = create_time_series(daily_yield, "batch_date", "yield_pct", "Daily Average Yield %")
    st.plotly_chart(fig, use_container_width=True)


def render_batch_explorer(batch_df, step_df):
    """Render the Batch Explorer page."""
    st.markdown('<p class="main-header">Batch Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore individual batch details and performance</p>', unsafe_allow_html=True)

    # Batch table
    st.subheader("Batch Summary Table")

    display_cols = ["batch_id", "batch_date", "product", "site", "line", "shift",
                    "yield_pct", "viability_pct", "potency_proxy", "impurity_pct", "passed_qc"]

    # Add color coding for pass/fail
    styled_df = batch_df[display_cols].copy()
    styled_df["batch_date"] = styled_df["batch_date"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        styled_df.style.apply(
            lambda x: ["background-color: #d4edda" if v else "background-color: #f8d7da"
                       for v in x] if x.name == "passed_qc" else [""] * len(x),
            axis=0
        ),
        height=400,
        use_container_width=True
    )

    st.markdown("---")

    # Batch detail selection
    st.subheader("Batch Detail View")
    selected_batch = st.selectbox("Select a batch to view details:", batch_df["batch_id"].tolist())

    if selected_batch:
        batch_info = batch_df[batch_df["batch_id"] == selected_batch].iloc[0]
        batch_steps = step_df[step_df["batch_id"] == selected_batch].sort_values("step_sequence")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Batch Metadata")
            st.write(f"**Batch ID:** {batch_info['batch_id']}")
            st.write(f"**Lot ID:** {batch_info['lot_id']}")
            st.write(f"**Date:** {batch_info['batch_date'].strftime('%Y-%m-%d')}")
            st.write(f"**Product:** {batch_info['product']}")
            st.write(f"**Site/Line:** {batch_info['site']} / {batch_info['line']}")
            st.write(f"**Shift:** {batch_info['shift']}")
            st.write(f"**Operator:** {batch_info['operator_id']}")

            status = "✅ PASSED" if batch_info["passed_qc"] else "❌ FAILED"
            st.markdown(f"**QC Status:** {status}")

            if batch_info.get("bad_lot_event"):
                st.warning("⚠️ Bad lot event detected")
            if batch_info.get("maintenance_event"):
                st.info("🔧 Post-maintenance batch")

        with col2:
            st.markdown("#### Process Step Metrics")
            fig = create_run_chart(
                batch_steps, "process_step", "yield_pct",
                f"Yield % by Process Step - {selected_batch}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Batch narrative
        st.markdown("---")
        st.markdown("#### Batch Narrative")

        narrative = generate_batch_narrative(batch_info, batch_steps, batch_df)
        st.markdown(narrative)


def generate_batch_narrative(batch_info, batch_steps, all_batches):
    """Generate plain-language narrative for a batch."""
    narratives = []

    # Overall assessment
    if batch_info["passed_qc"]:
        narratives.append("✅ **Overall:** This batch passed all QC specifications.")
    else:
        narratives.append("❌ **Overall:** This batch failed QC specifications.")

    # Yield comparison
    avg_yield = all_batches["yield_pct"].mean()
    yield_diff = batch_info["yield_pct"] - avg_yield
    if abs(yield_diff) > 5:
        direction = "above" if yield_diff > 0 else "below"
        narratives.append(f"📊 **Yield:** {batch_info['yield_pct']:.1f}% is {abs(yield_diff):.1f}% {direction} the overall average of {avg_yield:.1f}%.")
    else:
        narratives.append(f"📊 **Yield:** {batch_info['yield_pct']:.1f}% is within normal range (avg: {avg_yield:.1f}%).")

    # Impurity check
    if batch_info["impurity_pct"] > 3:
        narratives.append(f"⚠️ **Impurity:** Elevated impurity level ({batch_info['impurity_pct']:.2f}%) detected.")

    # Temperature excursions
    if batch_info["total_temp_excursions"] > 0:
        narratives.append(f"🌡️ **Temperature:** {batch_info['total_temp_excursions']} temperature excursion(s) recorded during processing.")

    # Deviations
    if batch_info["total_deviations"] > 0:
        narratives.append(f"📝 **Deviations:** {batch_info['total_deviations']} deviation(s) documented for this batch.")

    return "\n\n".join(narratives)


def render_spc_charts(batch_df, step_df):
    """Render the SPC / Control Charts page."""
    st.markdown('<p class="main-header">SPC / Control Charts</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Statistical process control for drift detection and monitoring</p>', unsafe_allow_html=True)

    # Chart configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        metric = st.selectbox("Select Metric", ["yield_pct", "viability_pct", "potency_proxy", "impurity_pct"])
    with col2:
        chart_type = st.selectbox("Chart Type", ["I-MR Chart", "EWMA Chart", "CUSUM Chart"])
    with col3:
        group_by = st.selectbox("Group By", ["All Data", "Site", "Product"])

    st.markdown("---")

    # Get data based on grouping
    if group_by == "All Data":
        groups = {"All": batch_df}
    elif group_by == "Site":
        groups = {site: batch_df[batch_df["site"] == site] for site in batch_df["site"].unique()}
    else:
        groups = {prod: batch_df[batch_df["product"] == prod] for prod in batch_df["product"].unique()}

    specs = get_specification_limits()
    metric_specs = specs.get(metric, {})

    for group_name, group_df in groups.items():
        if len(group_df) < 5:
            continue

        st.subheader(f"{group_name}")

        values = group_df.sort_values("batch_date")[metric].values
        dates = group_df.sort_values("batch_date")["batch_date"].dt.strftime("%m/%d").tolist()

        if chart_type == "I-MR Chart":
            i_result = calculate_individuals_chart(values)
            mr_result = calculate_moving_range_chart(values)
            fig = create_imr_chart(i_result, mr_result, f"I-MR Chart: {metric}", dates)
            st.plotly_chart(fig, use_container_width=True)

            # Alert summary
            if i_result.alerts:
                st.warning(f"⚠️ {len(i_result.alerts)} control chart violation(s) detected")
                alert_types = [a[1].value for a in i_result.alerts]
                st.caption(f"Alert types: {', '.join(set(alert_types))}")

        elif chart_type == "EWMA Chart":
            lambda_val = st.slider("EWMA λ (smoothing)", 0.1, 0.5, 0.2, key=f"ewma_{group_name}")
            target = metric_specs.get("target")
            result = calculate_ewma(values, lambda_param=lambda_val, target=target)
            fig = create_ewma_chart(result, f"EWMA Chart: {metric}", dates)
            st.plotly_chart(fig, use_container_width=True)

            if result.alerts:
                st.warning(f"⚠️ {len(result.alerts)} EWMA signal(s) detected - potential process drift")

        else:  # CUSUM
            target = metric_specs.get("target")
            result = calculate_cusum(values, target=target)
            fig = create_cusum_chart(result, f"CUSUM Chart: {metric}", dates)
            st.plotly_chart(fig, use_container_width=True)

            if result.alerts:
                st.warning(f"⚠️ {len(result.alerts)} CUSUM signal(s) detected - sustained shift from target")

        st.markdown("---")

    # Educational note
    with st.expander("ℹ️ Understanding Control Charts"):
        st.markdown("""
        **I-MR (Individuals and Moving Range) Chart:**
        - Best for low-volume manufacturing where individual measurements are taken
        - Upper chart shows individual values with 3σ control limits
        - Lower chart shows variation between consecutive points

        **EWMA (Exponentially Weighted Moving Average):**
        - More sensitive to small, sustained shifts in the process mean
        - λ parameter controls how much weight is given to recent observations
        - Time-varying control limits account for the evolving EWMA variance

        **CUSUM (Cumulative Sum):**
        - Accumulates deviations from target to detect sustained shifts
        - Positive CUSUM tracks upward shifts; negative tracks downward shifts
        - Signal when cumulative sum exceeds decision boundary (h)

        **Alert Rules Applied:**
        - Beyond 3σ: Any point outside control limits
        - 2 of 3 beyond 2σ: Two of three consecutive points beyond 2σ (same side)
        - Run of 8: Eight consecutive points on same side of centerline
        - Trend of 6: Six consecutive points trending in same direction
        """)


def render_capability(batch_df):
    """Render the Capability Analysis page."""
    st.markdown('<p class="main-header">Capability & Performance Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Process capability indices and specification conformance</p>', unsafe_allow_html=True)

    # Info box
    st.info("""
    **Note on Ppk vs Cpk:** This dashboard emphasizes **Ppk** (Process Performance Index) over Cpk.
    Ppk uses overall process variation and is more appropriate for assessing actual process performance,
    especially in low-N manufacturing. Cpk requires within-subgroup variation estimation and assumes
    a stable process, which may not always apply.
    """)

    specs = get_specification_limits()

    # Metric selection
    col1, col2 = st.columns([1, 2])
    with col1:
        metric = st.selectbox("Select Metric", list(specs.keys()))
        metric_spec = specs[metric]

    with col2:
        st.markdown(f"""
        **Specification Limits for {metric}:**
        - LSL: {metric_spec.get('LSL', 'N/A')}
        - Target: {metric_spec.get('target', 'N/A')}
        - USL: {metric_spec.get('USL', 'N/A')}
        """)

    st.markdown("---")

    # Overall capability
    st.subheader("Overall Process Capability")

    values = batch_df[metric].dropna().values
    result = calculate_capability(
        values,
        lsl=metric_spec.get("LSL"),
        usl=metric_spec.get("USL"),
        target=metric_spec.get("target")
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ppk_color = "normal" if result.ppk >= 1.33 else "inverse" if result.ppk >= 1.0 else "off"
        st.metric("Ppk", f"{result.ppk:.2f}", help="Process Performance Index")
    with col2:
        st.metric("Pp", f"{result.pp:.2f}", help="Process Performance")
    with col3:
        st.metric("Within Spec", f"{result.within_spec_pct:.1f}%")
    with col4:
        st.metric("Sample Size", f"{result.n_samples}")

    # Capability interpretation
    if result.ppk >= 1.33:
        st.success("✅ Process is capable (Ppk ≥ 1.33)")
    elif result.ppk >= 1.0:
        st.warning("⚠️ Process is marginally capable (1.0 ≤ Ppk < 1.33)")
    else:
        st.error("❌ Process is not capable (Ppk < 1.0)")

    # Histogram
    fig = create_capability_histogram(values, result, f"Capability Analysis: {metric}")
    st.plotly_chart(fig, use_container_width=True)

    # Tolerance interval
    st.markdown("---")
    st.subheader("Tolerance Interval")
    lower, upper = calculate_tolerance_interval(values, confidence=0.95, coverage=0.95)
    st.write(f"**95/95 Tolerance Interval:** [{lower:.2f}, {upper:.2f}]")
    st.caption("This interval is expected to contain 95% of the population with 95% confidence.")

    # Capability by group
    st.markdown("---")
    st.subheader("Capability by Product & Site")

    cap_results = []
    for product in batch_df["product"].unique():
        for site in batch_df["site"].unique():
            subset = batch_df[(batch_df["product"] == product) & (batch_df["site"] == site)]
            if len(subset) < 10:
                continue
            vals = subset[metric].dropna().values
            cap = calculate_capability(vals, metric_spec.get("LSL"), metric_spec.get("USL"), metric_spec.get("target"))
            cap_results.append({
                "Product": product,
                "Site": site,
                "n": cap.n_samples,
                "Mean": cap.mean,
                "Std": cap.std,
                "Ppk": cap.ppk,
                "Within Spec %": cap.within_spec_pct
            })

    if cap_results:
        cap_df = pd.DataFrame(cap_results)
        st.dataframe(
            cap_df.style.background_gradient(subset=["Ppk"], cmap="RdYlGn", vmin=0.5, vmax=2.0),
            hide_index=True,
            use_container_width=True
        )

        # Heatmap
        fig = create_heatmap(cap_df, "Site", "Product", "Ppk", "Ppk by Product and Site")
        st.plotly_chart(fig, use_container_width=True)


def render_trends_drivers(batch_df):
    """Render the Trends & Drivers page."""
    st.markdown('<p class="main-header">Trends & Drivers Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Time-series trends and key performance drivers</p>', unsafe_allow_html=True)

    # Time series trends
    st.subheader("📈 Metric Trends Over Time")

    metric_cols = ["yield_pct", "viability_pct", "potency_proxy", "impurity_pct"]
    selected_metrics = st.multiselect("Select metrics to analyze:", metric_cols, default=["yield_pct"])

    if selected_metrics:
        trend_results = analyze_time_trends(batch_df, "batch_date", selected_metrics, "site")

        # Show trend charts
        for metric in selected_metrics:
            daily_data = batch_df.groupby(["batch_date", "site"])[metric].mean().reset_index()
            import plotly.express as px
            fig = px.line(daily_data, x="batch_date", y=metric, color="site",
                          title=f"{metric} Trend by Site", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        # Trend summary table
        st.markdown("#### Trend Summary")
        trend_display = trend_results[["group", "metric", "trend", "pct_change_over_period", "interpretation"]].copy()
        trend_display.columns = ["Group", "Metric", "Trend", "% Change", "Interpretation"]
        st.dataframe(trend_display, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Driver analysis
    st.subheader("🔍 Driver Analysis")
    st.markdown("Identifying factors that influence yield performance")

    target = st.selectbox("Target metric:", ["yield_pct", "viability_pct", "potency_proxy"])
    features = ["total_hold_time_hrs", "total_cycle_time_hrs", "total_temp_excursions",
                "total_deviations", "impurity_pct"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Correlation Analysis")
        correlations = calculate_correlations(batch_df, target, features)

        if correlations:
            corr_df = pd.DataFrame([
                {"Feature": c.feature.replace("_", " ").title(),
                 "Correlation": c.correlation,
                 "Interpretation": c.interpretation}
                for c in correlations[:5]
            ])
            st.dataframe(corr_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Feature Importance (Random Forest)")
        drivers = analyze_drivers_regression(batch_df, target, features, model_type="forest")

        if drivers:
            fig = create_driver_importance_chart(drivers[:5], f"Top Drivers of {target}")
            st.plotly_chart(fig, use_container_width=True)

    # Driver interpretations
    st.markdown("---")
    st.markdown("#### Key Insights")

    if drivers:
        for i, driver in enumerate(drivers[:3], 1):
            st.markdown(f"**{i}. {driver.feature.replace('_', ' ').title()}:** {driver.interpretation}")

    # Group comparison
    st.markdown("---")
    st.subheader("📊 Group Comparisons")

    compare_by = st.selectbox("Compare by:", ["site", "product", "shift"])
    comparison = compare_groups(batch_df, "yield_pct", compare_by)

    if not comparison.empty:
        st.dataframe(
            comparison[["group", "n", "mean", "std", "diff_from_overall", "significant"]].rename(
                columns={"group": compare_by.title(), "n": "Count", "mean": "Mean Yield",
                         "std": "Std Dev", "diff_from_overall": "Diff from Avg", "significant": "Significant"}
            ),
            hide_index=True,
            use_container_width=True
        )


def render_about():
    """Render the About / Methodology page."""
    st.markdown('<p class="main-header">About This Dashboard</p>', unsafe_allow_html=True)

    st.markdown("""
    ## Purpose

    This is a **demonstration application** showcasing manufacturing analytics and
    statistical process control (SPC) capabilities. It is designed to illustrate
    how modern analytics tools can support process monitoring, quality control,
    and continuous improvement in manufacturing environments.

    **⚠️ Important:** This dashboard uses **entirely synthetic data** and is not
    connected to any real manufacturing processes. It is intended for educational
    and demonstration purposes only.

    ---

    ## Synthetic Data Generation

    The data is generated using a fixed random seed for reproducibility. Key characteristics:

    ### Dimensions
    - **Products:** 4 biopharmaceutical products with different baseline performance
    - **Sites:** 3 manufacturing sites (Boston, Dublin, Singapore)
    - **Process Steps:** 7 steps from Thaw & Seed through QC Release
    - **Time Period:** Full year (2024)

    ### Built-in Patterns
    - **Site effects:** Dublin site has simulated equipment drift over time
    - **Shift effects:** Night shift has slightly higher variability
    - **Correlations:** Hold time negatively correlates with yield
    - **Events:** Random bad lot events (~2%) and maintenance resets (~1%)

    ### Metrics Generated
    - Yield %, Viability %, Potency proxy
    - Impurity %, Cycle time, Hold time
    - Temperature excursions, Deviations

    ---

    ## Methodology

    ### Control Charts
    - **I-MR Charts:** Individuals and Moving Range charts suitable for low-volume manufacturing
    - **EWMA:** Exponentially Weighted Moving Average for detecting small sustained shifts
    - **CUSUM:** Cumulative Sum charts for sustained deviation detection
    - **Western Electric Rules:** Standard run rules for alert generation

    ### Capability Analysis
    - **Ppk (Process Performance Index):** Uses overall standard deviation - recommended for
      assessing actual process performance
    - **Cpk (Process Capability Index):** Uses within-subgroup variation - shown with disclaimer
      about assumptions
    - **Tolerance Intervals:** 95/95 intervals computed using normal distribution factors

    ### Driver Analysis
    - **Correlation Analysis:** Pearson correlations with significance testing
    - **Feature Importance:** Random Forest regression for non-linear relationships
    - **Trend Analysis:** Linear regression on time series with slope significance testing

    ---

    ## Technology Stack

    - **Streamlit:** Web application framework
    - **Plotly:** Interactive visualizations
    - **Pandas/NumPy:** Data manipulation
    - **Scikit-learn:** Machine learning models
    - **SciPy:** Statistical functions

    ---

    ## Source Code

    This application is open source. The synthetic data generator, SPC calculations,
    and visualization components are modular and can be adapted for other use cases.

    ---

    ## Disclaimer

    This is a demonstration application and should not be used for actual manufacturing
    decisions. Real manufacturing analytics systems require:

    - Validated data pipelines from actual equipment
    - Regulatory compliance considerations (FDA, EMA, etc.)
    - Proper statistical validation of methods
    - Integration with quality management systems
    - Security and access controls

    For production use, consult with qualified statisticians and quality professionals.
    """)


def main():
    """Main application entry point."""
    # Load data
    step_df, batch_df = load_data()

    # Render sidebar and get selections
    page, products, sites, date_range, shifts = render_sidebar()

    # Apply filters
    filtered_batch = apply_filters(batch_df, products, sites, date_range, shifts)
    filtered_step = apply_filters(step_df, products, sites, date_range, shifts)

    # Render selected page
    if page == "📊 Overview":
        render_overview(filtered_batch, filtered_step)
    elif page == "🔍 Batch Explorer":
        render_batch_explorer(filtered_batch, filtered_step)
    elif page == "📈 SPC / Control Charts":
        render_spc_charts(filtered_batch, filtered_step)
    elif page == "🎯 Capability Analysis":
        render_capability(filtered_batch)
    elif page == "📉 Trends & Drivers":
        render_trends_drivers(filtered_batch)
    elif page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()
