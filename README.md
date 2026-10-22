# PanBrida

BRD4 Paper 2 research application (Streamlit).

## Features
- Single and batch SMILES input
- RDKit Morgan fingerprint generation
- Saved QSAR model inference (requires a fitted model artifact)
- Chemical validity, simple descriptors and candidate CSV export

## Deployment
1. Deploy this repository on Streamlit Community Cloud, entrypoint `app.py`.
2. Supply a trusted, fitted QSAR model artifact at `models/brd4_model.joblib` (or upload it in the app for the current session).
3. Confirm the model's training fingerprint parameters. Defaults in this app are radius 2 and 2048 bits **only as placeholders**; set these to the exact training settings before using predictions.

No training or model validation is claimed by this repository. Do not upload untrusted joblib/pickle artifacts, which can execute code when loaded.
