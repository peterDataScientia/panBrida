import io
import torch
from torch import nn
from rdkit import Chem

class SMILESRNN(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, layers):
        super().__init__()
        self.vocab=vocab
        self.embedding=nn.Embedding(len(vocab),embedding_size)
        self.rnn=nn.GRU(embedding_size,hidden_size,layers,batch_first=True)
        self.head=nn.Linear(hidden_size,len(vocab))
    def forward(self, ids, state=None):
        out,state=self.rnn(self.embedding(ids),state)
        return self.head(out),state

def load_generator(raw):
    # Accept a checkpoint path (CI/local), raw bytes (Streamlit upload), or file-like object.
    if isinstance(raw, (str, __import__("os").PathLike)):
        source = raw
    elif isinstance(raw, (bytes, bytearray)):
        source = io.BytesIO(raw)
    else:
        source = raw
    ck=torch.load(source,map_location="cpu",weights_only=True)
    vocab=ck["vocab"]
    if not all(x in vocab for x in ("^","$")):
        raise ValueError("Checkpoint vocabulary missing start/end tokens")
    model=SMILESRNN(vocab,ck["embedding_size"],ck["hidden_size"],ck["layers"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model

@torch.inference_mode()
def generate(model, count=40, max_length=120, temperature=1.0, seed=2026):
    torch.manual_seed(int(seed))
    vocab=model.vocab
    start=vocab.index("^"); stop=vocab.index("$")
    pad=vocab.index("_") if "_" in vocab else -1
    idx=torch.full((count,1),start,dtype=torch.long)
    ended=torch.zeros(count,dtype=torch.bool)
    state=None
    strings=[[] for _ in range(count)]
    for _ in range(max_length):
        logits,state=model(idx,state)
        logits=logits[:,-1,:].clone()/float(temperature)
        logits[:,start]=-1e9
        if pad>=0:logits[:,pad]=-1e9
        probs=torch.softmax(logits,dim=-1)
        nxt=torch.multinomial(probs,1).squeeze(1)
        for i,t in enumerate(nxt.tolist()):
            if not ended[i] and t!=stop:strings[i].append(vocab[t])
        ended |= (nxt==stop)
        idx=nxt[:,None]
        if bool(ended.all()):break
    seen=set();out=[]
    stats={"requested":int(count),"unfinished":0,"invalid":0,"duplicates":0,"unique_valid":0}
    for i,s in enumerate(strings):
        if not ended[i]:
            stats["unfinished"]+=1
            continue
        mol=Chem.MolFromSmiles("".join(s))
        if mol is None:
            stats["invalid"]+=1
            continue
        smi=Chem.MolToSmiles(mol,isomericSmiles=True)
        if smi not in seen:
            seen.add(smi);out.append((f"RNN_{len(out)+1:03d}",smi))
        else:
            stats["duplicates"]+=1
    stats["unique_valid"]=len(out)
    return out,stats
