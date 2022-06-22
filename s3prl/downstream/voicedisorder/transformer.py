import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

low_mem = True

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., layer_ind=0):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.layer_ind=layer_ind        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1) 
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class TransformerLayers(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for d in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, layer_ind=d))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            if low_mem:
                x = checkpoint(attn,x)
            else:
                x = attn(x)
            x = ff(x)
        return x
                                                              
class Transformer(nn.Module):
    def __init__(self, input_dim, 
                        output_dim,
                        hidden_dim=256, mlp_dim_factor=2,
                        dim_head=32, 
                        heads=6,
                        depth=2,                          
                        dropout=0.1):
        super().__init__()        
        
        self.fci = nn.Linear(input_dim, hidden_dim)
      
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.block = TransformerLayers(  hidden_dim, depth, heads, dim_head, hidden_dim*mlp_dim_factor, dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, x_len):
        b, n, _ = x.shape
        
        x = self.fci(x)

        x = torch.cat(( self.cls_token.repeat(b, 1, 1), x), dim=1)
        
        x = self.block(x)  
        tokenclass = x[:,0]
        predicted = self.out(tokenclass)
        
        return predicted, tokenclass      
        
   
## -----------------------------------------------------------------------------
if __name__ == '__main__':
    print('net example')
    tr = Transformer(300, 256)
    x = torch.randn(32, 100, 300)
    
    x1 = tr(x)
    print(x1.shape)
    
    
