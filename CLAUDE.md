# manufacturing-process-analytics

## Local Run
- Create a virtual environment and install `requirements.txt`
- Start the app with `streamlit run app.py`

## Hetzner Deploy
- Production host path: `/opt/manufacturing-process-analytics`
- Public URL: `https://mfg.norangio.dev`
- Reverse proxy target: `127.0.0.1:8502`
- systemd service: `manufacturing-process-analytics`
- GitHub Actions deploys on pushes to `main` using `VPS_HOST`, `VPS_USER`, and `VPS_SSH_KEY`
