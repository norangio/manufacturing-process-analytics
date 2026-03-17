# Manufacturing Analytics & SPC Dashboard

A professional demonstration application showcasing manufacturing process monitoring, statistical process control (SPC), and capability analysis using entirely synthetic data.

## 🎯 Purpose

This dashboard demonstrates how modern analytics tools can support:
- Real-time process monitoring and KPI tracking
- Statistical process control with multiple chart types
- Process capability and performance analysis
- Root cause and driver identification
- Trend analysis and drift detection

**⚠️ Note:** This uses synthetic data only and is for demonstration/educational purposes.

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/manufacturing-process-analytics.git
cd manufacturing-process-analytics

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Hetzner VPS Deployment

This repo can deploy to the same Hetzner server pattern used by the other `norangio.dev` apps. Running two Streamlit apps on the same box is fine as long as each one has its own port and systemd service. This app is configured for:

- Domain: `mfg.norangio.dev`
- App directory: `/opt/manufacturing-process-analytics`
- systemd service: `manufacturing-process-analytics`
- Streamlit bind address: `127.0.0.1:8502`

#### Required GitHub Actions secrets

- `VPS_HOST`
- `VPS_USER`
- `VPS_SSH_KEY`

#### One-time VPS setup

1. Add the DNS record:
   - `mfg.norangio.dev -> <your-vps-ip>`
2. Add the Caddy block from `deploy/Caddyfile.snippet` to `/etc/caddy/Caddyfile`
3. Reload Caddy:
   - `sudo systemctl reload caddy`

After that, pushing to `main` will deploy the app through `.github/workflows/deploy.yml`.

## 📊 Dashboard Pages

### 1. Overview
- High-level KPIs: batch counts, yield, pass rates
- Yield distribution by product
- Pass rate by site
- SPC alert summary
- Yield trend over time

### 2. Batch Explorer
- Searchable batch table with key metrics
- Detailed batch view with metadata
- Process step run charts
- Automated batch narrative generation

### 3. SPC / Control Charts
- **I-MR Charts:** Individuals and Moving Range for low-volume manufacturing
- **EWMA Charts:** Exponentially Weighted Moving Average for small shift detection
- **CUSUM Charts:** Cumulative Sum for sustained shift detection
- Western Electric run rules with alert highlighting

### 4. Capability Analysis
- Process Performance Index (Ppk) - emphasized for low-N manufacturing
- Process Capability Index (Cpk) - with appropriate disclaimers
- Capability histograms with specification limits
- Tolerance intervals (95/95)
- Capability comparison by product and site

### 5. Trends & Drivers
- Time-series trends with regression analysis
- Correlation analysis between metrics
- Feature importance using Random Forest
- Group comparisons with statistical significance

### 6. About / Methodology
- Documentation of methods used
- Synthetic data generation details
- Important disclaimers

## 🔧 Project Structure

```
manufacturing-process-analytics/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── AGENTS.md             # Local project/deploy notes
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions VPS deploy
├── deploy/
│   ├── server-deploy.sh  # Runs on the VPS
│   ├── manufacturing-process-analytics.service
│   └── Caddyfile.snippet
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── generator.py   # Synthetic data generation
    ├── analytics/
    │   ├── __init__.py
    │   ├── spc.py        # SPC calculations (I-MR, EWMA, CUSUM)
    │   └── drivers.py    # Driver analysis and trends
    └── viz/
        ├── __init__.py
        └── charts.py     # Plotly visualization components
```

## 📈 Synthetic Data

Data is generated with a fixed random seed (42) for reproducibility.

### Dimensions
- **Products:** BioProduct-A, BioProduct-B, CellTherapy-X, GeneVector-Z
- **Sites:** Boston, Dublin, Singapore
- **Process Steps:** 7 steps (Thaw & Seed → QC Release)
- **Time Period:** January - December 2024
- **Batches:** 500 batches with ~3,500 step-level records

### Built-in Patterns
- Product-specific baseline performance
- Site effects (Dublin has simulated drift)
- Shift variability (Night shift higher variance)
- Hold time → yield correlation
- Random events: bad lots (~2%), maintenance resets (~1%)

### Metrics
| Metric | LSL | Target | USL |
|--------|-----|--------|-----|
| Yield % | 60 | 85 | 100 |
| Viability % | 70 | 92 | 100 |
| Potency Proxy | 80 | 100 | 120 |
| Impurity % | 0 | 1.5 | 5 |

## 🛠️ Technology Stack

- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **SciPy** - Statistical functions
- **Scikit-learn** - Machine learning (driver analysis)

## 📋 SPC Methods

### Control Charts
- **Individuals Chart (I):** For individual measurements, control limits at ±3σ using moving range estimation
- **Moving Range Chart (MR):** Monitors process variation between consecutive measurements
- **EWMA:** Smoothed average with time-varying control limits, λ configurable
- **CUSUM:** Cumulative sum with configurable allowance (k) and decision interval (h)

### Western Electric Rules
1. Any point beyond 3σ
2. 2 of 3 consecutive points beyond 2σ (same side)
3. 4 of 5 consecutive points beyond 1σ (same side)
4. 8 consecutive points on same side of centerline
5. 6 consecutive points trending up or down

### Capability Indices
- **Ppk:** Uses overall standard deviation - recommended for performance assessment
- **Cpk:** Uses within-subgroup variation - requires stable process assumption
- **Tolerance Intervals:** Normal distribution-based 95/95 intervals

## ⚠️ Disclaimer

This is a **demonstration application** using synthetic data. It should NOT be used for actual manufacturing decisions. Production systems require:

- Validated data from real equipment
- Regulatory compliance (FDA, EMA, etc.)
- Statistical method validation
- QMS integration
- Proper security controls

## 📄 License

MIT License - See LICENSE file for details.
