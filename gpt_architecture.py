#import libs
import torch
import torch.nn as nn
from torch.nn import functional as F

#set random seed to make our result reproduceable
torch.manual_seed(1337)

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 250
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_heads = 6
n_layers = 6
dropout=0.2


#prepare the text
with open('input.txt', encoding="utf-8") as f:
  text = f.read()
#all the unqiue chars in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create a mapping from characters to numbers
stoi = {vocab:index for index, vocab in enumerate(chars)}
itos = {index:vocab for index, vocab in enumerate(chars)}
encode = lambda s: [stoi[i] for i in s]
decode = lambda c: ''.join([itos[i] for i in c])

#train/test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#load data
def get_batch(split):
  data = train_data if split=="train" else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
   out = {}
   model.eval()
   for split in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            #model.forward -> model
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split]=losses.mean()
   model.train()
   return out

class Head(nn.Module):
   """One head of self attention"""
   def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
      self.dropout = nn.Dropout(dropout)
      #buffer means the non parameter tensors in a model

   def forward(self, x):
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)
      wei = q @ k.transpose(-2,-1) * C**(-0.5)
      wei = wei.masked_fill(self.tril==0, float('-inf'))
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)
      v = self.value(x)
      out = wei @ v
      return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
     super().__init__()
     self.heads = nn.ModuleList([Head(head_size) for I in range(num_heads)])
     self.proj = nn.Linear(n_embd, n_embd)
     self.dropout = nn.Dropout(dropout)

  def forward(self,x):
     x = torch.cat([h(x) for h in self.heads], dim=-1)
     out = self.proj(x)
     out = self.dropout(out)
     return out
      
class FeedForward(nn.Module):
   def __init__(self, n_embd):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_embd, 4 * n_embd),
         nn.ReLU(),
         nn.Linear(4 * n_embd, n_embd),
         nn.Dropout(dropout)
      )
   def forward(self, x):
      return self.net(x)
   
#define the model
class GPT(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table=nn.Embedding(vocab_size, n_embd)
    self.positional_embedding_table = nn.Embedding(block_size, n_embd)
    self.block = nn.Sequential(*[Block(n_heads=n_heads, head_size=n_embd//n_heads, n_embd=n_embd) for _ in range(n_layers)])
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.block(x)
    logits = self.lm_head(x)
    
    if targets is None:
      loss=None
    else:
      B, T, C = logits.shape
      logits_view = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits_view, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss=self.forward(idx_cond)
      #only focus on the last time
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx

class Block(nn.Module):
  def __init__(self, n_heads, head_size, n_embd):
    super().__init__()
    self.MultiHeadAttention = MultiHeadAttention(n_heads, head_size)
    self.FeedForward = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)


  def forward(self, x):
    x = x + self.MultiHeadAttention(self.ln1(x))
    x = x + self.FeedForward(self.ln2(x))
    return x

#instaniate the model
model = GPT()
m = model.to(device)

#create the optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

#train and eval the model
for iter in range(max_iters):
    if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate text
context = torch.ones((1,500), dtype=torch.long).to(device)
print(decode(m.generate(context, 500)[0].tolist()))
