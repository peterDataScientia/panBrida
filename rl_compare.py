"""Reproducible paired pretrained-versus-REINFORCE benchmark."""
import json
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator
from rnn_inference import load_generator, generate
from rl_finetune import reinforce

torch.set_num_threads(2)
reference = pd.read_csv("data/BRD4_modeling_cohort.csv")["isomeric_smiles"].dropna().astype(str).tolist()
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
known = set()
fps_ref = []
for s in reference:
    mol = Chem.MolFromSmiles(s)
    if mol:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical not in known:
            known.add(canonical)
            fps_ref.append(fpgen.GetFingerprint(mol))

def evaluate(model, seed, n=300):
    molecules, stats = generate(model, count=n, temperature=0.8, seed=seed)
    rewards = []
    for _, smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        sim = max(DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(mol), fps_ref))
        novelty = max(0.0, 1.0 - sim)
        props = float(150 <= Descriptors.MolWt(mol) <= 650 and Descriptors.MolLogP(mol) <= 5)
        rewards.append(0.45 * novelty + 0.35 * QED.qed(mol) + 0.20 * props)
    return {"validity": (stats["unique_valid"] + stats["duplicates"]) / n,
            "unique_yield": stats["unique_valid"] / n,
            "mean_reward_valid": float(np.mean(rewards)) if rewards else 0.0,
            "unique_valid": stats["unique_valid"]}

base = load_generator("models/BRD4_generated.trained.pt")
tuned, history = reinforce(base, reference, steps=30, batch_size=32, max_length=100, seed=2026)
rows = []
for seed in (2027, 2028, 2029):
    for label, agent in (("Pretrained", base), ("REINFORCE", tuned)):
        rows.append({"seed": seed, "model": label, **evaluate(agent, seed)})
frame = pd.DataFrame(rows)
frame.to_csv("rl_comparison.csv", index=False)
pd.DataFrame(history).to_csv("rl_training_history.csv", index=False)
summary = frame.groupby("model")[["validity", "unique_yield", "mean_reward_valid"]].agg(["mean", "std"])
print(summary.to_string())
paired = frame.pivot(index="seed", columns="model", values=["validity", "unique_yield", "mean_reward_valid"])
for metric in ("validity", "unique_yield", "mean_reward_valid"):
    delta = paired[(metric, "REINFORCE")] - paired[(metric, "Pretrained")]
    print(f"{metric} paired delta mean={delta.mean():.4f} sd={delta.std(ddof=1):.4f}")
print("Comparison complete. No automatic claim of RL improvement.")
