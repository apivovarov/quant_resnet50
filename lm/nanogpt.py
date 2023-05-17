import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import time
print(torch.__version__)
print(f"{torch.cuda.is_available()=}")
device="cpu"
if torch.cuda.is_available():
    device="cuda:0"
print(device)

names_f = "tinyshakespeare/input.txt"
with open(names_f) as f:
    text = f.read()

#random.seed(42)
print(text[:30])
print(f"{len(text)=}")

chars = sorted(set(text))
voc_size = len(chars)
print(f"{chars[:100]=}")
print(f"{voc_size=}")

itos = dict()
stoi = dict()
for i, c in enumerate(chars):
    itos[i] = c
    stoi[c] = i
def encode(ss):
    return [stoi[c] for c in ss]
def decode(ii):
    return ''.join([itos[i] for i in ii])
print(encode("Hello\nWorld"))
print(decode(encode("Hello\nWorld")))

data = torch.tensor(encode(text), dtype=torch.long, device=device)
print(f"{data.shape=}")
print(data[:30])
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


block_size = 256
head_size = 64
n_heads = 6
emb_size = head_size * n_heads
n_layers = 6
dropout_rate = 0.2
batch_size = 64
learning_rate = 1e-3
use_bias = False
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type="cuda", dtype=torch.float16)

def get_batch(data, batch_size, device):
    ix = torch.randint(low=0,high=len(data)-block_size, size=(batch_size,), device=device)
    x = torch.stack([data[i : i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix]).to(device)
    return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_size, 3 * emb_size, bias=use_bias)
        # output projection
        self.c_proj = nn.Linear(emb_size, emb_size, bias=use_bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.n_heads = n_heads
        self.head_size = head_size
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.scale = head_size ** -0.5
        # flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        else:
            print("Using Flash Attention!")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.emb_size, dim=2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_rate if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc    = nn.Linear(emb_size, emb_size*4, bias=use_bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(emb_size*4, emb_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_heads) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(emb_size)
        self.mlp = MLP()
    def forward(self, x):
        # x shape N,BL,EMB
        x = x + self.attn(self.ln1(x)) # N,BL,Hxn_heads
        x = x + self.mlp(self.ln2(x)) # N,BL,Hxn_heads # FeedForward
        return x


class NanoGptModel(nn.Module):
    def __init__(self, voc_size) -> None:
        super().__init__()
        self.wte = nn.Embedding(voc_size, emb_size)
        self.wpe = nn.Embedding(block_size, emb_size)
        self.drop = nn.Dropout(dropout_rate)
        self.blocks = nn.Sequential(*[DecoderBlock(n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, voc_size, bias=False)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                print("!!!", pn)
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

    def forward(self, ids):
        B, T = ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.wte(ids)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.lm_head(x)
        return x

    def calc_loss(self, logits, Y):
        logits = logits.transpose(1,2)
        return F.cross_entropy(logits, Y)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens):
        for i in range(max_new_tokens):
            logits = self(ids[:,-block_size:])
            logits = logits[:,-1,:]
            prob = torch.softmax(logits, dim=-1)
            y = torch.multinomial(prob, num_samples=1)
            ids = torch.cat((ids, y), dim=-1)
        return ids.detach().cpu().numpy()

@torch.no_grad()
def estimate_loss(N):
    model.eval()
    WIN=[]
    for _ in range(N):
        x, y_target = get_batch(val_data, batch_size, device)
        with ctx:
            logits = model(x)
            loss = model.calc_loss(logits, y_target)
        WIN.append(loss.item())
    model.train()
    return np.mean(WIN)

# head = DotProductAttn(is_mask=True)
# mha = MultiHeadAttn(n_heads=n_heads, is_mask=True)
# mha = CausalSelfAttention()
# dec_block0 = DecoderBlock(n_heads)
# dec_block1 = DecoderBlock(n_heads)
# x = torch.randn((1, block_size, emb_size), dtype=torch.float32)
# out = mha(x)
# print(out.shape)

model = NanoGptModel(voc_size=voc_size).to(device)
n_of_params = sum([p.numel() for p in model.parameters()])
print(f"Numel: {n_of_params:,}")

x, y = get_batch(train_data, batch_size, device)
out = model(x)
print(out.shape)
loss = model.calc_loss(out, y)
print("loss:", loss.item())

res = model.generate(x, 4)
print(res.shape)
print(decode(res[0]))

iter = 0
losst, lossv, lri = [],[],[]

# Training Loop
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
learning_rate = 1e-3

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
WIN = []
N = 100
first = True
t0 = time.time()
for i in range(iter, iter + N):
    if not first:
        lr = get_lr(i)
        lri.append(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    x, y_target = get_batch(train_data, batch_size, device)
    logits = model(x)
    loss = model.calc_loss(logits, y_target)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    WIN.append(loss.item())
    if (i+1) % 10 == 0:
        avg_loss = np.mean(WIN)
        WIN=[]
        losst.append(avg_loss)
        print(f"{i+1}: losst: {losst[-1]:.4f}, lr: {lr:.6f}, time: {dt*1000:.2f} ms")
    if (i+1) % 250 == 0:
        loss_est = estimate_loss(10)
        lossv.append(loss_est)
        print(f"{i+1}: lossv: {lossv[-1]:.4f}")
    iter += 1
    first = False

if len(losst) > 0 and len(lossv) > 0:
    print(f"losst: {losst[-1]:.4f}, lossv: {lossv[-1]:.4f}")


# Validation Loop
val_loss = estimate_loss(20)
print(f"{val_loss=}")

x, y = get_batch(val_data, 1, device)
model.eval()
#x = torch.zeros((1,block_size), dtype=torch.long, device=device)
res = model.generate(x, 500)
print(decode(res[0,:block_size]))
print("--<START>--")
print(decode(res[0,block_size:]))

torch.save(model.state_dict(), "gpt-state-dict.pt")
