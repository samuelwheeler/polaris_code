import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pos_encoding import PositionalEncoding
import math

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Causal_Kron_Block(nn.Module):
    def __init__(self, dim_in, max_len, heads):
        assert dim_in % heads == 0
        super().__init__()
        self.max_len = max_len
        self.heads = heads
        self.mat1 = nn.Linear(dim_in, dim_in, bias = False)
        self.mat1.weight = nn.Parameter(torch.randn(dim_in, dim_in) * (1/dim_in))
        # self.mat2 = nn.Parameter(torch.randn(heads, max_len, max_len) * (1/max_len))
        self.mat2 = nn.Parameter(torch.randn(heads, max_len, max_len) * (torch.ones(max_len,max_len) * torch.tensor([1/k for k in range(1, max_len+1)])).T)  
        self.w_out = nn.Parameter(torch.randn(heads, dim_in // heads, dim_in) * (1/dim_in))

    def forward(self, x):
        x_len = x.shape[1]
        assert x_len <= self.max_len
        x = self.mat1(x)
        x = rearrange(x, 'b l (h d) -> b h l d', h = self.heads)
        x = torch.matmul(torch.tril(self.mat2[:, :x_len, :x_len]), x)
        x = torch.matmul(x, self.w_out)
        x = torch.sum(x, dim = 1)
        return x


class Causal_Kron_Block_MLP(nn.Module):
    def __init__(self, dim_in, max_len, heads):
        assert dim_in % heads == 0
        super().__init__()
        self.max_len = max_len
        self.heads = heads
        self.mat1a = nn.Linear(dim_in, dim_in, bias = False)
        self.mat1a.weight = nn.Parameter(torch.randn(dim_in, dim_in) * (1/dim_in))
        self.mat1b = nn.Linear(dim_in, dim_in, bias = False)
        self.mat1b.weight = nn.Parameter(torch.randn(dim_in, dim_in) * (1/dim_in))
        # self.mat2 = nn.Parameter(torch.randn(heads, max_len, max_len) * (1/max_len))
        self.mat2a = nn.Parameter(torch.randn(heads, max_len, max_len) * (torch.ones(max_len,max_len) * torch.tensor([1/k for k in range(1, max_len+1)])).T)
        self.mat2b = nn.Parameter(torch.randn(heads, max_len, max_len) * (torch.ones(max_len,max_len) * torch.tensor([1/k for k in range(1, max_len+1)])).T)    
        self.w_out = nn.Parameter(torch.randn(heads, dim_in // heads, dim_in) * (1/dim_in))
        self.relu = nn.ReLU()

    def forward(self, x):
        x_len = x.shape[1]
        assert x_len <= self.max_len
        x = self.mat1a(x)
        x = rearrange(x, 'b l (h d) -> b h l d', h = self.heads)
        x = torch.matmul(torch.tril(self.mat2a[:, :x_len, :x_len]), x)
        x = self.relu(x)
        x = rearrange(x, 'b h l d -> b l (h d)',  h = self.heads)
        x = self.mat1b(x)
        x = rearrange(x, 'b l (h d) -> b h l d', h = self.heads)
        x = torch.matmul(torch.tril(self.mat2b[:, :x_len, :x_len]), x)
        x = torch.matmul(x, self.w_out)
        x = torch.sum(x, dim = 1)
        return x

class MLP_Mixer_Block(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        self.l1 = nn.Linear(max_len, max_len)
        self.l2 = nn.Linear(max_len, max_len)
        self.relu = nn.ReLU()
    def forward(self, x):
        x_len = x.shape[1]
        assert x_len <= self.max_len
        x = rearrange(x, 'b l d -> b d l')
        x = F.linear(x, torch.tril(self.l1.weight[:, :x_len]), self.l1.bias[:x_len])
        x = self.relu(x)
        x = F.linear(x, torch.tril(self.l2.weight[:, :x_len]), self.l2.bias[:x_len])
        x = rearrange(x, 'b d l -> b l d')
        return x


class Feedforward(nn.Module):
    def __init__(self, dim_in, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim_in)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
class FlatSum(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.linear = PreNorm(d_model, Feedforward(d_model, hidden_dim))
    def forward(self,x):
        x = torch.cumsum(x, 1)
        x = self.linear(x)
        return x

class Proxy_Mixer(nn.Module):
    def __init__(self, d_model, num_proxies):
        super().__init__()
        self.proxy = nn.Parameter(torch.randn(num_proxies, d_model))
        self.softmax1 = torch.nn.Softmax(dim = 2)
        self.softmax2 = torch.nn.Softmax(dim = 1)
        self.num_proxies = num_proxies
    def forward(self, x):
        seq_len = x.shape[1]
        prods = torch.matmul(self.proxy, x.transpose(-1,1))
        print(prods.shape)
        x = prods[:,:,:,None] * x[:,None,:]
        x = torch.cumsum(x, dim = 2)
        # x /= ((x.new_ones(seq_len).cumsum(0)+1)**0.5)[:,None]
        x = self.softmax2(prods[:,:,:,None]/(self.num_proxies**0.5)) * x
        x = torch.sum(x, dim = 1)
        return x

class KronMixer(nn.Module):
    def __init__(self, *, max_len, d_model, hidden_dim, vocab_size, depth, heads, dropout, kron_type = None, num_proxies = 100):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model = d_model, dropout = dropout)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        layers = []
        for d in range(depth):
            if kron_type == 'Kron_MLP':
                layers.append(PreNorm(d_model, Causal_Kron_Block_MLP(d_model, max_len, heads)))
            elif kron_type == 'standard': 
                layers.append(PreNorm(d_model, Causal_Kron_Block(d_model, max_len, heads)))
            elif kron_type == 'MLP_Mixer':
                layers.append(PreNorm(d_model, MLP_Mixer_Block(max_len)))
            elif kron_type == 'Flat':
                layers.append(FlatSum(d_model, hidden_dim))
            elif kron_type == 'Proxy':
                layers.append(PreNorm(d_model, Proxy_Mixer(d_model, num_proxies)))
            layers.append(PreNorm(d_model, Feedforward(d_model, hidden_dim)))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(d_model, vocab_size)
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
           
    def forward(self, x, verbose = False):
        x = self.embed(x)
        x = self.pos_encoder(x)
        if verbose:
            print('X SHAPE: ', x.shape)
        for layer in self.layers:
            if isinstance(layer, Causal_Kron_Block):
                x = layer(x, verbose) + x
            x = layer(x) + x
        return self.linear(x)
