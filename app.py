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
st.info("PanBrida loads its bundled generator and BRD4 reference cohort automatically. QSAR potency prediction is enabled only when a compatible validated model is bundled.")

with st.sidebar:
    st.header("QSAR configuration")
    metadata_path = Path("models/brd4_metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    radius = int(metadata.get("radius", 2))
    n_bits = int(metadata.get("n_bits", 1024))
    st.caption(f"QSAR fingerprint configuration: Morgan radius {radius}, {n_bits} bits.")
    model = None
    model_path = Path("models/brd4_model.joblib")
    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception as exc:
            st.error(f"Bundled QSAR model failed to load: {exc}")
    else:
        st.caption("QSAR model not bundled; descriptor screening and RNN generation remain available.")
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
    st.caption("Optional external CSV screening. No upload is needed for built-in BRD4 generation and novelty.")
    csv = st.file_uploader("External CSV (optional)", type=["csv"], key="csv")
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
    st.caption("The trained generator checkpoint is bundled and loaded automatically.")
    rnn_count = st.number_input("RNN sampling batch size", min_value=10, max_value=10000, value=1000, step=100, help="Larger batches take longer; generation and novelty calculations may be resource-intensive.")
    rnn_temp = st.slider("Sampling temperature", 0.5, 1.5, 0.8, 0.05)
    st.caption("Default 0.8 is a hypothesis for improving validity, not yet benchmarked. Compare raw validity and diversity across temperatures.")
    local_checkpoint = Path("models/BRD4_generated.trained.pt")
    if not local_checkpoint.exists():
        st.error("Bundled generator checkpoint is missing from deployment.")
    if st.button("Generate with trained RNN", disabled=not local_checkpoint.exists()):
        try:
            from rnn_inference import load_generator, generate
            generator = load_generator(local_checkpoint)
            with st.spinner("Sampling trained BRD4 generator..."):
                candidates, generation_stats = generate(generator, count=rnn_count, temperature=rnn_temp, seed=2026)
            st.session_state["rnn_generated"] = screen(candidates)
            st.session_state["rnn_stats"] = generation_stats
            st.success(f"{len(candidates)} unique valid molecules generated from {rnn_count} samples.")
        except Exception as exc:
            st.error(f"RNN inference failed: {exc}")
    st.divider()
    st.subheader("Reinforcement learning (REINFORCE)")
    st.caption("Fine-tune a fresh copy of the trained GRU using policy-gradient updates. No manual model upload. Research prototype; reward does not establish BRD4 activity.")
    rl_steps = st.number_input("RL training steps", min_value=1, max_value=100, value=10, step=1)
    rl_batch = st.number_input("RL trajectories per step", min_value=4, max_value=128, value=32, step=4)
    if st.button("Train reward-guided RNN", disabled=not local_checkpoint.exists()):
        try:
            from rnn_inference import load_generator
            from rl_finetune import reinforce
            reference_path_rl = Path("data/BRD4_modeling_cohort.csv")
            if not reference_path_rl.exists():
                st.error("Bundled BRD4 reference missing; RL training cannot proceed.")
            else:
                frame_rl = pd.read_csv(reference_path_rl)
                col_rl = next(c for c in frame_rl.columns if "smiles" in c.lower())
                with st.spinner("Running policy-gradient updates on CPU..."):
                    tuned, history = reinforce(load_generator(local_checkpoint), frame_rl[col_rl].dropna().astype(str).tolist(), steps=int(rl_steps), batch_size=int(rl_batch))
                st.session_state["rl_agent"] = tuned
                st.session_state["rl_history"] = pd.DataFrame(history)
                st.success("REINFORCE training completed. Use the fine-tuned agent to generate molecules.")
        except Exception as exc:
            st.error(f"RL training failed: {exc}")
    if "rl_history" in st.session_state:
        st.dataframe(st.session_state["rl_history"], hide_index=True, use_container_width=True)
        st.line_chart(st.session_state["rl_history"].set_index("Step")[["Mean reward", "Valid fraction"]])
        if st.button("Generate with RL fine-tuned RNN"):
            try:
                from rnn_inference import generate
                molecules_rl, stats_rl = generate(st.session_state["rl_agent"], count=int(rnn_count), temperature=float(rnn_temp), seed=2027)
                st.session_state["rnn_generated"] = screen(molecules_rl)
                st.session_state["rnn_stats"] = stats_rl
                st.success(f"RL agent produced {len(molecules_rl)} unique valid candidates.")
            except Exception as exc:
                st.error(f"RL generation failed: {exc}")
    if "rnn_generated" in st.session_state:
        stats = st.session_state.get("rnn_stats")
        if stats:
            st.write("Raw generation audit")
            st.json(stats)
            st.metric("Unique-valid yield", f"{100*stats['unique_valid']/stats['requested']:.1f}%")
            st.download_button("Download generation audit", json.dumps(stats, indent=2), "panbrida_generation_audit.json", "application/json")
        generated_df = st.session_state["rnn_generated"].copy()
        if not generated_df.empty:
            st.subheader("Reward-guided candidate selection")
            st.caption("Post-generation reward scoring and selection (not reinforcement-learning updates to RNN weights).")
            valid_mask = generated_df["Status"].eq("Valid")
            if valid_mask.any():
                reference_path_reward = Path("data/BRD4_modeling_cohort.csv")
                known_reward = set()
                ref_fps_reward = []
                fpgen_reward = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                if reference_path_reward.exists():
                    reference_reward = pd.read_csv(reference_path_reward)
                    ref_col = next((c for c in reference_reward.columns if "smiles" in c.lower()), None)
                    if ref_col:
                        for smi in reference_reward[ref_col].dropna().astype(str):
                            mol = Chem.MolFromSmiles(smi)
                            if mol:
                                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                                if canonical not in known_reward:
                                    known_reward.add(canonical)
                                    ref_fps_reward.append(fpgen_reward.GetFingerprint(mol))
                valid_rows = generated_df.loc[valid_mask].copy()
                mols_reward = [Chem.MolFromSmiles(smi) for smi in valid_rows["SMILES"]]
                nearest_reward = [max(DataStructs.BulkTanimotoSimilarity(fpgen_reward.GetFingerprint(mol), ref_fps_reward)) if ref_fps_reward else np.nan for mol in mols_reward]
                valid_rows["Nearest_reference_similarity"] = nearest_reward
                valid_rows["Exact_novel"] = [smi not in known_reward for smi in valid_rows["SMILES"]]
                valid_rows["Novelty_reward"] = [max(0.0, 1.0 - v) if np.isfinite(v) else 0.0 for v in nearest_reward]
                valid_rows["QED_reward"] = valid_rows["QED"].astype(float).clip(0, 1)
                valid_rows["Property_reward"] = [float(150 <= Descriptors.MolWt(mol) <= 650 and rdMolDescriptors.CalcTPSA(mol) <= 140 and Descriptors.MolLogP(mol) <= 5) for mol in mols_reward]
                potency_available = model is not None and "Predicted model output" in valid_rows.columns
                if potency_available:
                    predictions_reward = pd.to_numeric(valid_rows["Predicted model output"], errors="coerce")
                    potency_available = predictions_reward.notna().all()
                if potency_available:
                    valid_rows["Potency_reward"] = (1 / (1 + np.exp(-np.clip((predictions_reward - 6.0), -30, 30)))).to_numpy()
                    valid_rows["Reward"] = 0.40 * valid_rows["Potency_reward"] + 0.25 * valid_rows["Novelty_reward"] + 0.20 * valid_rows["QED_reward"] + 0.15 * valid_rows["Property_reward"]
                    st.caption("Reward = 0.40 potency + 0.25 fingerprint novelty + 0.20 QED + 0.15 property compliance. Potency is model-predicted, not experimental.")
                else:
                    valid_rows["Reward"] = 0.45 * valid_rows["Novelty_reward"] + 0.35 * valid_rows["QED_reward"] + 0.20 * valid_rows["Property_reward"]
                    st.warning("QSAR potency model unavailable. Reward uses novelty, QED and property compliance only; it is NOT BRD4 potency optimization.")
                valid_rows = valid_rows.sort_values("Reward", ascending=False)
                st.dataframe(valid_rows, use_container_width=True, hide_index=True)
                st.download_button("Download reward-ranked candidates", valid_rows.to_csv(index=False).encode(), "panbrida_reward_ranked.csv", "text/csv")
        valid_smiles = generated_df.loc[generated_df["Status"] == "Valid", "SMILES"].tolist()
        if len(valid_smiles) > 1:
            fps = [rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(Chem.MolFromSmiles(s)) for s in valid_smiles]
            similarities = []
            for i in range(len(fps)):
                similarities.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
            if similarities:
                st.metric("Mean pairwise Tanimoto similarity (Morgan2048)", f"{np.mean(similarities):.3f}")
                st.caption("Lower similarity suggests greater fingerprint diversity; this is not evidence of BRD4 activity.")
        reference_path = Path("data/BRD4_modeling_cohort.csv")
        reference = reference_path if reference_path.exists() else None
        if reference is None:
            st.error("Bundled BRD4 reference cohort missing; novelty audit unavailable.")
        if reference is not None and valid_smiles:
            try:
                training = pd.read_csv(reference)
                columns = [c for c in training.columns if "smiles" in c.lower()]
                if not columns:
                    st.error("Training CSV must contain a SMILES column.")
                else:
                    known = set()
                    for smi in training[columns[0]].dropna().astype(str):
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            known.add(Chem.MolToSmiles(mol, isomericSmiles=True))
                    novel = sum(s not in known for s in valid_smiles)
                    st.metric("Novel versus BRD4 reference cohort", f"{novel}/{len(valid_smiles)} ({100*novel/len(valid_smiles):.1f}%)")
                    st.caption(f"Compared with {len(known)} distinct valid reference structures; exact canonical-SMILES novelty only.")
                    if known:
                        fpgen_ref = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                        ref_fps = [fpgen_ref.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in sorted(known)]
                        audit_rows = []
                        for smi in valid_smiles:
                            mol = Chem.MolFromSmiles(smi)
                            if mol is None:
                                continue
                            fp = fpgen_ref.GetFingerprint(mol)
                            max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
                            audit_rows.append({"SMILES": smi, "Exact_novel": smi not in known,
                                               "Nearest_training_Tanimoto_Morgan2048": round(max_sim, 4),
                                               "Similarity_below_0.8": max_sim < 0.8,
                                               "Similarity_below_0.6": max_sim < 0.6})
                        novelty_df = pd.DataFrame(audit_rows)
                        st.metric("Novel with nearest similarity < 0.8",
                                  f"{int((novelty_df['Exact_novel'] & novelty_df['Similarity_below_0.8']).sum())}/{len(novelty_df)}")
                        st.metric("Novel with nearest similarity < 0.6",
                                  f"{int((novelty_df['Exact_novel'] & novelty_df['Similarity_below_0.6']).sum())}/{len(novelty_df)}")
                        st.dataframe(novelty_df, use_container_width=True)
                        st.download_button("Download full novelty audit CSV",
                                           novelty_df.to_csv(index=False).encode("utf-8"),
                                           "panbrida_novelty_audit.csv", "text/csv")
                        st.caption("Nearest-neighbor Tanimoto is a fingerprint similarity measure, not proof of chemical novelty or biological activity.")
            except Exception as exc:
                st.error(f"Reference dataset audit failed: {exc}")
        show_results(st.session_state["rnn_generated"])
    st.divider()
    st.subheader("Temperature benchmark")
    st.caption("Compare 0.8, 0.9 and 1.0 using equal sample counts and independent seeds. No retraining.")
    benchmark_n = st.slider("Samples per replicate", 100, 500, 300, 50, key="benchmark_n")
    benchmark_reps = st.slider("Replicates per temperature", 2, 10, 3, key="benchmark_reps")
    if st.button("Run temperature benchmark", disabled=not local_checkpoint.exists()):
        try:
            from rnn_inference import load_generator, generate
            model_rnn = load_generator(local_checkpoint)
            rows_bench = []
            fpgen_bench = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            progress = st.progress(0)
            total = 3 * benchmark_reps
            for j, temperature in enumerate((0.8, 0.9, 1.0)):
                for rep in range(benchmark_reps):
                    seed = 2026 + rep
                    molecules, counts = generate(model_rnn, count=benchmark_n, temperature=temperature, seed=seed)
                    fps = [fpgen_bench.GetFingerprint(Chem.MolFromSmiles(smi)) for _, smi in molecules]
                    sim_sum, sim_n = 0.0, 0
                    for k, fp in enumerate(fps[:-1]):
                        sims = DataStructs.BulkTanimotoSimilarity(fp, fps[k+1:])
                        sim_sum += sum(sims)
                        sim_n += len(sims)
                    rows_bench.append({
                        "Temperature": temperature, "Replicate": rep + 1, "Seed": seed,
                        **counts,
                        "Raw validity %": 100 * (counts["unique_valid"] + counts["duplicates"]) / counts["requested"],
                        "Unique-valid yield %": 100 * counts["unique_valid"] / counts["requested"],
                        "Mean pairwise Tanimoto": sim_sum / sim_n if sim_n else np.nan
                    })
                    progress.progress((j * benchmark_reps + rep + 1) / total)
            st.session_state["temperature_benchmark"] = pd.DataFrame(rows_bench)
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")
    if "temperature_benchmark" in st.session_state:
        bench = st.session_state["temperature_benchmark"]
        st.dataframe(bench, hide_index=True, use_container_width=True)
        summary = bench.groupby("Temperature")[["Raw validity %", "Unique-valid yield %", "Mean pairwise Tanimoto"]].agg(["mean", "std"])
        st.subheader("Mean and SD across seeds")
        st.dataframe(summary, use_container_width=True)
        st.caption("Prefer high unique-valid yield and low similarity; inspect trade-offs rather than choosing a temperature from validity alone.")
        st.download_button("Download benchmark CSV", bench.to_csv(index=False).encode(),
                           "panbrida_temperature_benchmark.csv", "text/csv")
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
