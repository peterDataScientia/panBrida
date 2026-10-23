"""Bounded REINFORCE fine-tuning of the bundled SMILES GRU.
Reward uses QED, novelty against reference, and physicochemical constraints.
This is a research prototype: no claims of experimental BRD4 potency.
"""
import copy
import numpy as np
import torch
from torch.distributions import Categorical
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, rdFingerprintGenerator

def reinforce(model, reference_smiles, steps=10, batch_size=32, lr=0.00005, temperature=1.0, seed=2026, max_length=120):
    torch.manual_seed(int(seed))
    agent = copy.deepcopy(model).cpu()
    agent.train()
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(lr))
    vocab = agent.vocab
    start, stop = vocab.index("^"), vocab.index("$")
    pad = vocab.index("_") if "_" in vocab else -1
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    reference = set()
    reference_fps = []
    for s in reference_smiles:
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canonical not in reference:
                reference.add(canonical)
                reference_fps.append(fpgen.GetFingerprint(mol))
    history = []
    baseline = 0.0
    for step in range(int(steps)):
        idx = torch.full((batch_size, 1), start, dtype=torch.long)
        ended = torch.zeros(batch_size, dtype=torch.bool)
        state = None
        logprob = torch.zeros(batch_size)
        sequences = [[] for _ in range(batch_size)]
        for _ in range(max_length):
            logits, state = agent(idx, state)
            scores = logits[:, -1, :] / float(temperature)
            scores = scores.clone()
            scores[:, start] = -1e9
            if pad >= 0:
                scores[:, pad] = -1e9
            dist = Categorical(logits=scores)
            token = dist.sample()
            active = ~ended
            logprob = logprob + dist.log_prob(token) * active.float()
            for i, t in enumerate(token.tolist()):
                if active[i] and t != stop:
                    sequences[i].append(vocab[t])
            ended = ended | (token == stop)
            idx = token[:, None]
            if bool(ended.all()):
                break
        rewards = []
        valid = 0
        for i, chars in enumerate(sequences):
            mol = Chem.MolFromSmiles("".join(chars)) if ended[i] else None
            if mol is None:
                rewards.append(0.0)
                continue
            valid += 1
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            similarity = max(DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(mol), reference_fps)) if reference_fps else 1.0
            novelty = max(0.0, 1.0 - similarity)
            properties = float(150 <= Descriptors.MolWt(mol) <= 650 and rdMolDescriptors.CalcTPSA(mol) <= 140 and Descriptors.MolLogP(mol) <= 5)
            rewards.append(0.45 * novelty + 0.35 * QED.qed(mol) + 0.20 * properties)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = float(reward_tensor.mean())
        advantage = reward_tensor - baseline
        loss = -(advantage.detach() * logprob).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        baseline = 0.9 * baseline + 0.1 * mean_reward if step else mean_reward
        history.append({"Step": step + 1, "Mean reward": mean_reward, "Valid fraction": valid / batch_size, "Loss": float(loss.detach())})
    agent.eval()
    return agent, history
