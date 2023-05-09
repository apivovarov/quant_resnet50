import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

device = "cpu"
voc_size=27
att = 3
emb = 3
hidden = 100

E_w = torch.rand(voc_size, emb, device=device)
H0_w = torch.rand(size=(emb*att, hidden), device=device)
H0_b = torch.rand(size=(hidden,), device=device)
H1_w = torch.rand(size=(hidden, voc_size), device=device)
H1_b = torch.rand(size=(voc_size,), device=device)
params = [E_w, H0_w, H0_b, H1_w, H1_b]
nparams = sum([t.numel() for t in params])
print(f"{nparams=}")

def forward(X):
    Ey = E_w[X].flatten(1, 2)
    H0_y = torch.tanh(Ey @ H0_w + H0_b)
    L = H0_y @ H1_w + H1_b
    L = torch.softmax(L, dim=-1)
    return L

X = torch.tensor([[1,2,3]])
Y = forward(X)
print(Y)

traced = torch.jit.trace(forward, X)
print("Tracing Done")
torch.onnx.export(
    model=traced, 
    args=X,
    f="lm1.onnx",
    verbose=True
)
print("ONNX export is done")