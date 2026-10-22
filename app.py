from pathlib import Path
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, rdFingerprintGenerator

st.set_page_config(page_title="PanBrida | BRD4 Discovery", page_icon="🧬", layout="wide")
st.title("🧬 PanBrida")
st.caption("BRD4 Paper 2 | Structure screening, QSAR inference and candidate prioritization")
st.warning("Research prototype. No validated BRD4 model is bundled; predictions require a trusted, fitted artifact and matching training settings.")

with st.sidebar:
    st.header("QSAR configuration")
    model_file = st.file_uploader("Trusted trained model (.joblib)", type=["joblib"], key="model")
    metadata_file = st.file_uploader("Training metadata (.json)", type=["json"])
    metadata = {}
    if metadata_file:
        try:
            metadata = json.loads(metadata_file.getvalue().decode("utf-8"))
            st.success("Metadata loaded")
        except Exception as exc:
            st.error(f"Invalid metadata: {exc}")
    radius = st.number_input("Morgan radius", 1, 4, int(metadata.get("radius", 2)))
    n_bits = st.number_input("Morgan bits", 128, 8192, int(metadata.get("n_bits", 2048)), step=128)
    st.caption("Confirm these values against the original training pipeline.")
    st.markdown("**Model output:** pIC50 only if the saved model was trained on pIC50.")
    model = None
    if model_file:
        try:
            model = joblib.load(io.BytesIO(model_file.getvalue()))
        except Exception as exc:
            st.error(f"Model loading failed: {exc}")
    elif Path("models/brd4_model.joblib").exists():
        try:
            model = joblib.load("models/brd4_model.joblib")
        except Exception as exc:
            st.error(f"Bundled model failed: {exc}")
    st.caption("Only load artifacts you trust: joblib can execute code.")
    st.divider()
    st.write("**Model loaded:**", "Yes" if model is not None else "No")
    st.write("**Applicability domain:**", "Not yet calibrated")

single, batch, about = st.tabs(["SMILES screening", "Batch CSV", "Methodology"])

def parse_lines(raw):
    items = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        items.append((parts[1] if len(parts) > 1 else f"BRD4_{i:03d}", parts[0]))
    return items

def screen(items):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    rows, vectors, positions = [], [], []
    for compound_id, smiles in items:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            rows.append({"ID": compound_id, "SMILES": smiles, "Status": "Invalid SMILES"})
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        row = {
            "ID": compound_id, "SMILES": canonical, "Status": "Valid",
            "MW": round(Descriptors.MolWt(mol), 2),
            "cLogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "QED": round(QED.qed(mol), 3),
        }
        fp = generator.GetFingerprint(mol)
        vector = np.zeros((int(n_bits),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, vector)
        positions.append(len(rows))
        vectors.append(vector)
        rows.append(row)
    if model is not None and vectors:
        try:
            x = np.vstack(vectors)
            if hasattr(model, "n_features_in_") and int(model.n_features_in_) != x.shape[1]:
                raise ValueError(f"Model expects {model.n_features_in_} features, but fingerprint has {x.shape[1]}")
            predictions = np.asarray(model.predict(x)).reshape(-1)
            if len(predictions) != len(positions):
                raise ValueError("Unexpected number of model predictions")
            for position, value in zip(positions, predictions):
                rows[position]["Predicted model output"] = round(float(value), 4)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    return pd.DataFrame(rows)

def show_results(df):
    if df.empty:
        st.info("No candidates supplied.")
        return
    a, b, c = st.columns(3)
    a.metric("Candidates", len(df))
    b.metric("Valid", int((df["Status"] == "Valid").sum()))
    c.metric("Invalid", int((df["Status"] != "Valid").sum()))
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("Download screening CSV", df.to_csv(index=False).encode(), "panbrida_screening.csv", "text/csv")
    valid = df[df["Status"] == "Valid"]
    if not valid.empty:
        st.download_button("Download PanDoc candidate list", "\n".join(f"{r.SMILES} {r.ID}" for r in valid.itertuples(index=False)) + "\n", "panbrida_pandoc.txt", "text/plain")
    if "Predicted model output" in df.columns:
        st.caption("Model outputs have not been independently validated in this app. Do not interpret as pIC50 unless the uploaded model provenance confirms its target.")

with single:
    st.write("Enter one SMILES per line, optionally followed by a compound ID.")
    raw = st.text_area("SMILES list", height=220, placeholder="CCO Test_01\nCC(=O)O Test_02")
    if st.button("Screen SMILES", type="primary"):
        st.session_state["results"] = screen(parse_lines(raw))
    if "results" in st.session_state:
        show_results(st.session_state["results"])

with batch:
    csv = st.file_uploader("Upload CSV with a SMILES column", type=["csv"], key="csv")
    if csv:
        try:
            frame = pd.read_csv(csv)
            st.caption(f"{len(frame)} input rows")
            smiles_col = st.selectbox("SMILES column", list(frame.columns), index=next((i for i,c in enumerate(frame.columns) if c.lower() in ("smiles", "canonical_smiles")), 0))
            id_col = st.selectbox("ID column", ["(generate IDs)"] + list(frame.columns))
            if st.button("Screen CSV", type="primary"):
                items = [(str(frame.iloc[i][id_col]) if id_col != "(generate IDs)" else f"BRD4_{i+1:03d}", str(frame.iloc[i][smiles_col])) for i in range(len(frame))]
                st.session_state["batch_results"] = screen(items)
        except Exception as exc:
            st.error(f"CSV error: {exc}")
    if "batch_results" in st.session_state:
        show_results(st.session_state["batch_results"])

with about:
    st.subheader("Scientific provenance")
    st.write("Molecules are standardized to RDKit canonical isomeric SMILES. Morgan fingerprints are calculated using RDKit's Morgan generator. Descriptors and QED are computed directly from molecular structures.")
    st.write("For reproducible QSAR inference, supply the exact fitted estimator, fingerprint radius, bit length, chirality settings, RDKit version, target transformation, and training-set provenance.")
    st.write("Applicability domain, XAI/SHAP and generative-model inference are pending integration with their original fitted artifacts. They are intentionally not simulated.")
    st.write("PanBrida is a computational research tool, not an experimental activity assay.")
