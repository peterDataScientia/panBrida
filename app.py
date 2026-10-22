from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors

st.set_page_config(page_title="PanBrida | BRD4 Research", page_icon="🧬", layout="wide")
st.title("PanBrida")
st.caption("BRD4 Paper 2 · Explainable QSAR and candidate screening · Research use only")
st.info("QSAR predictions require your **previously trained model** and the **exact same fingerprint settings** used during training. No model is bundled.")

with st.sidebar:
    st.header("Prediction settings")
    radius = st.number_input("Morgan radius (verify against training)", 1, 4, 2)
    n_bits = st.number_input("Morgan fingerprint bits (verify against training)", 128, 8192, 2048, step=128)
    st.caption("Defaults are placeholders, NOT verified BRD4 training parameters.")
    uploaded = st.file_uploader("Load trusted trained model (.joblib)", type=["joblib"])
    st.warning("Only load model files you trust. Joblib can execute Python code.")
    model_path = Path("models/brd4_model.joblib")
    model = None
    if uploaded is not None:
        try:
            model = joblib.load(io.BytesIO(uploaded.getvalue()))
        except Exception as exc:
            st.error(f"Could not load model: {exc}")
    elif model_path.is_file():
        try:
            model = joblib.load(model_path)
        except Exception as exc:
            st.error(f"Could not load bundled model: {exc}")

st.subheader("Candidate structures")
st.caption("One SMILES per line, optionally followed by a space and compound ID (PanDoc-compatible).")
raw = st.text_area("SMILES and optional ID", height=220, placeholder="CCO Example_01\nCC(=O)O Example_02")
if st.button("Analyze candidates", type="primary"):
    records = []
    fingerprints = []
    valid_positions = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        smiles = parts[0]
        compound_id = parts[1] if len(parts) == 2 else f"Candidate_{line_number:03d}"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            records.append({"ID": compound_id, "SMILES": smiles, "Status": "Invalid SMILES"})
            continue
        row = {
            "ID": compound_id,
            "SMILES": Chem.MolToSmiles(mol, isomericSmiles=True),
            "Status": "Valid",
            "MW": round(Descriptors.MolWt(mol), 2),
            "cLogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "QED": round(QED.qed(mol), 3),
        }
        valid_positions.append(len(records))
        records.append(row)
        fingerprint = AllChem.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits)).GetFingerprint(mol)
        fingerprints.append(np.asarray(list(fingerprint), dtype=np.float32))
    if not records:
        st.warning("Enter at least one candidate.")
    else:
        if model is not None and fingerprints:
            try:
                values = np.asarray(model.predict(np.vstack(fingerprints)), dtype=float).reshape(-1)
                if len(values) != len(valid_positions):
                    raise ValueError("Model returned unexpected prediction count")
                for position, prediction in zip(valid_positions, values):
                    records[position]["Predicted pIC50"] = round(float(prediction), 4)
                st.warning("Predictions are provisional until model provenance, feature representation, and applicability domain are verified.")
            except Exception as exc:
                st.error(f"Model prediction failed: {exc}")
        elif model is None:
            st.info("No trained model loaded: showing structure screening only. Predicted potency is intentionally omitted.")
        results = pd.DataFrame(records)
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download results CSV", results.to_csv(index=False).encode("utf-8"), file_name="panbrida_candidates.csv", mime="text/csv")

st.divider()
st.caption("PanBrida does not claim experimental BRD4 inhibition, docking validation, or calibrated potency estimates.")
