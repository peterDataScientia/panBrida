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
st.info("Trained BRD4 SMILES-RNN generation is supported. Load the saved checkpoint in the Generate tab; QSAR predictions require the separately trained SVR artifact.")

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
    n_bits = st.number_input("Morgan bits", 128, 8192, int(metadata.get("n_bits", 1024)), step=128)
    st.caption("The BRD4 Colab notebook references an SVR–Morgan1024 artifact; verify radius and chirality from the saved training metadata.")
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

single, batch, generation, about = st.tabs(["SMILES screening", "Batch CSV", "Generate molecules", "Methodology"])

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
            estimator = model
            if isinstance(model, dict):
                estimator = next((model[k] for k in ("model", "estimator", "svr", "regressor", "pipeline") if k in model and hasattr(model[k], "predict")), None)
                if estimator is None:
                    raise ValueError("Model bundle is a dictionary without a recognized fitted estimator; inspect its keys and training notebook")
            x = np.vstack(vectors)
            if hasattr(estimator, "n_features_in_") and int(estimator.n_features_in_) != x.shape[1]:
                raise ValueError(f"Model expects {estimator.n_features_in_} features, but fingerprint has {x.shape[1]}")
            predictions = np.asarray(estimator.predict(x)).reshape(-1)
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


with generation:
    st.subheader("Trained BRD4 SMILES-RNN")
    st.caption("Verified checkpoint format: 37-token vocabulary, 128-dimensional embeddings, 256 hidden units, 2 GRU layers. No retraining required.")
    st.markdown("[Get the trained checkpoint from your connected Google Drive](https://drive.google.com/file/d/1XqSL8l2t4qn_VZUCa3MuX9DmnXEwELMj/view)")
    checkpoint = st.file_uploader("Upload BRD4_generated.trained.pt", type=["pt"], key="rnn_checkpoint")
    rnn_count = st.slider("RNN sampling batch size", 10, 300, 100, 10)
    rnn_temp = st.slider("Sampling temperature", 0.5, 1.5, 1.0, 0.05)
    local_checkpoint = Path("models/BRD4_generated.trained.pt")
    if not local_checkpoint.exists() and checkpoint is None:
        st.info("Select the checkpoint from the Drive link above. The trained checkpoint is not yet bundled in this GitHub repository.")
    if st.button("Generate with trained RNN", disabled=checkpoint is None and not local_checkpoint.exists()):
        try:
            from rnn_inference import load_generator, generate
            generator = load_generator(checkpoint.getvalue() if checkpoint is not None else local_checkpoint.read_bytes())
            with st.spinner("Sampling trained BRD4 generator..."):
                candidates, generation_stats = generate(generator, count=rnn_count, temperature=rnn_temp, seed=2026)
            st.session_state["rnn_generated"] = screen(candidates)
            st.session_state["rnn_stats"] = generation_stats
            st.success(f"{len(candidates)} unique valid molecules generated from {rnn_count} samples.")
        except Exception as exc:
            st.error(f"RNN inference failed: {exc}")
    if "rnn_generated" in st.session_state:
        stats = st.session_state.get("rnn_stats")
        if stats:
            st.write("Raw generation audit")
            st.json(stats)
            st.metric("Unique-valid yield", f"{100*stats['unique_valid']/stats['requested']:.1f}%")
            st.download_button("Download generation audit", json.dumps(stats, indent=2), "panbrida_generation_audit.json", "application/json")
        show_results(st.session_state["rnn_generated"])
    st.divider()
    st.subheader("Structure-based analogue generation")
    st.caption("RDKit chemical edits on supplied seed structures. This is NOT the trained SMILES-RNN generator.")
    seed_text = st.text_area("Seed SMILES (one per line, optional ID)", height=140, key="seed_input", placeholder="CCOc1ccccc1 Seed_01")
    n_target = st.slider("Maximum unique generated candidates", 5, 200, 40, step=5)
    random_seed = st.number_input("Random seed", min_value=0, value=2026, step=1)
    mw_limit = st.slider("Maximum molecular weight", 250, 900, 650, step=25)
    qed_min = st.slider("Minimum QED", 0.0, 1.0, 0.3, step=0.05)
    st.caption("Operations: replace a terminal halogen or alkyl substituent, append a small terminal substituent to available aromatic carbon, or mutate terminal halogens. Products are sanitized and deduplicated.")
    if st.button("Generate analogues", type="primary"):
        import random
        rng = random.Random(int(random_seed))
        seeds = parse_lines(seed_text)
        edits = ["F", "Cl", "C", "O", "N"]
        products, seen = [], set()
        valid_seeds = []
        for _, smi in seeds:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                seen.add(canonical)
                valid_seeds.append(mol)
        if not valid_seeds:
            st.error("Provide at least one valid seed SMILES.")
        else:
            attempts = 0
            while len(products) < n_target and attempts < max(500, n_target * 120):
                attempts += 1
                parent = rng.choice(valid_seeds)
                rw = Chem.RWMol(parent)
                eligible = [a.GetIdx() for a in parent.GetAtoms() if a.GetSymbol() == "C" and a.GetIsAromatic() and a.GetTotalNumHs() > 0]
                terminals = [a.GetIdx() for a in parent.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I", "C", "O", "N") and a.GetDegree() == 1 and not a.GetIsAromatic()]
                if rng.random() < 0.65 and eligible:
                    idx = rng.choice(eligible)
                    new_idx = rw.AddAtom(Chem.Atom(rng.choice(edits)))
                    rw.AddBond(idx, new_idx, Chem.BondType.SINGLE)
                elif terminals:
                    idx = rng.choice(terminals)
                    original = parent.GetAtomWithIdx(idx).GetSymbol()
                    choices = [x for x in edits if x != original]
                    rw.GetAtomWithIdx(idx).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(rng.choice(choices)))
                else:
                    continue
                try:
                    product = rw.GetMol()
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product, isomericSmiles=True)
                    if smi in seen or Descriptors.MolWt(product) > mw_limit or QED.qed(product) < qed_min:
                        continue
                    seen.add(smi)
                    products.append((f"PanBrida_{len(products)+1:03d}", smi))
                except Exception:
                    continue
            if len(products) < n_target:
                st.info(f"Generated {len(products)} unique valid analogues after {attempts} attempts; filters or seed substitution sites limited diversity.")
            st.session_state["generated"] = screen(products)
            st.session_state["generation_attempts"] = attempts
    if "generated" in st.session_state:
        generated = st.session_state["generated"]
        if not generated.empty:
            if "Predicted model output" in generated.columns:
                generated = generated.sort_values("Predicted model output", ascending=False)
            else:
                generated = generated.sort_values("QED", ascending=False)
            show_results(generated)
            st.caption("Ranking is by predicted model output only when a model is loaded; otherwise by QED. A high QED is not evidence of BRD4 activity.")

with about:
    st.subheader("Scientific provenance")
    st.write("Molecules are standardized to RDKit canonical isomeric SMILES. Morgan fingerprints are calculated using RDKit's Morgan generator. Descriptors and QED are computed directly from molecular structures.")
    st.write("For reproducible QSAR inference, supply the exact fitted estimator, fingerprint radius, bit length, chirality settings, RDKit version, target transformation, and training-set provenance.")
    st.write("RDKit analogue generation is available. The previously trained SMILES-RNN generator, applicability domain and XAI/SHAP remain pending integration with their original fitted artifacts. They are intentionally not simulated.")
    st.write("PanBrida is a computational research tool, not an experimental activity assay.")
