import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import einops

class SSM(nn.Module):
    def __init__(self, 
                 d_expanded: int,
                 d_rank: int,
                 d_hidden: int = 16,
                 dt_min: float = 1e-3,
                 dt_max: float = 1e-1,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_expanded = d_expanded
        self.d_rank = d_rank
        self.d_hidden = d_hidden
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.device = device

        # projection for delta, B, C
        self.x_proj = nn.Linear(d_expanded, d_rank + 2*d_hidden, device=device)
        
        # broadcast for delta
        self.dt_proj = nn.Linear(d_rank, d_expanded, device=device)
        dt_init_std = 1/(d_rank)**(1/2)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
    
        dt = torch.exp(
            torch.rand(self.d_expanded) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # initialization of A
        A = einops.repeat(torch.arange(1, self.d_hidden+1), 'n -> d n', d=self.d_expanded).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

    def forward(self, x: torch.Tensor):
        """
        x: torch.Tensor (B, L, d_expanded)
        """
        dt, B, C = torch.split(self.x_proj(x), [self.d_rank, self.d_hidden, self.d_hidden], dim=-1)
        A = -torch.exp(self.A_log) # (d_expanded, d_hidden)
        dt = F.linear(dt, self.dt_proj.weight)  # (B L d_expanded)
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype)) # (B L d_expanded)
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A)) # (B L d_expanded d_hidden)
        dB = torch.einsum("bld,bln->bldn", dt, B) # (B L d_expanded d_hidden)
        dBX = torch.einsum("bldn,bld->bldn", dB, x) # (B L d_expanded d_hidden)
        for i in range(int(math.log2(x.shape[1]))):
            dBX[:, 2**i:] = dA[:, 2**i:]*dBX[:, :-2**i] + dBX[:, 2**i:]
            dA[:, 2**i:] = dA[:, 2**i:]*dA[:, :-2**i]
        
        y = torch.einsum("bldn,bln->bld", dBX, C)
        return y


class MambaBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 expansion: int = 2,
                 d_rank: int = None,
                 d_hidden: int = 16,
                 dt_min: float = 1e-3,
                 dt_max: float = 1e-1,
                 kernel_size: int = 4,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.expansion = expansion
        self.d_rank = d_rank if d_rank is not None else math.ceil(d_model / 16)
        self.d_hidden = d_hidden
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.kernel_size = kernel_size
        self.device = device
        self.act = nn.SiLU()
        self.d_expanded = int(d_model * expansion)
        
        self.in_proj = nn.Linear(d_model, 2*self.d_expanded, device=device)
        self.out_proj = nn.Linear(self.d_expanded, d_model, device=device)

        self.conv1d = nn.Conv1d(self.d_expanded, self.d_expanded, self.kernel_size, padding=self.kernel_size-1, bias=True, groups=self.d_expanded, device=device)
        self.ssm = SSM(self.d_expanded, self.d_rank, self.d_hidden, self.dt_min, self.dt_max, self.device)

    def forward(self, x: torch.Tensor):
        """
        x: torch.Tensor (B, L, d_model)
        """
        x, z = torch.split(self.in_proj(x), [self.d_expanded, self.d_expanded], dim=-1)
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.ssm(x)
        x = x*self.act(z)
        x = self.out_proj(x)
        return x
    
class MambaLNBlock(nn.Module):
    def __init__(self,
                    d_model: int,
                    expansion: int = 2,
                    d_rank: int = None,
                    d_hidden: int = 16,
                    dt_min: float = 1e-3,
                    dt_max: float = 1e-1,
                    kernel_size: int = 4,
                    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    *args, 
                    **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.d_model = d_model
            self.expansion = expansion
            self.d_rank = d_rank
            self.d_hidden = d_hidden
            self.dt_min = dt_min
            self.dt_max = dt_max
            self.kernel_size = kernel_size
            self.device = device
            
            self.ln = nn.LayerNorm(d_model, device=device)
            self.mamba = MambaBlock(d_model, expansion, d_rank, d_hidden, dt_min, dt_max, kernel_size, device)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None):
        """
        x: torch.Tensor (B, L, d_model)
        residual: torch.Tensor (B, L, d_model)
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.ln(hidden_states)
        hidden_states = self.mamba(hidden_states)
        return hidden_states, residual


class MambaLLM(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 d_model: int = 512,
                 n_layers: int = 10,
                 expansion: int = 2,
                 d_rank: int = None,
                 d_hidden: int = 16,
                 dt_min: float = 1e-3,
                 dt_max: float = 1e-1,
                 kernel_size: int = 4,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.n_layers = n_layers
        self.expansion = expansion
        self.d_rank = d_rank
        self.d_hidden = d_hidden
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.kernel_size = kernel_size
        self.device = device

        self.token_emb = nn.Embedding(num_tokens, d_model, device=device)
        self.blocks = nn.ModuleList([
            MambaLNBlock(d_model, expansion, d_rank, d_hidden, dt_min, dt_max, kernel_size, device)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model, device=device)
        self.head = nn.Linear(d_model, num_tokens, bias=False, device=device)

    def forward(self, x: torch.Tensor):
        """
        x: torch.Tensor (B, L)
        """
        residual = None
        x = self.token_emb(x)
        for block in self.blocks:
            x, residual = block(x)
        residual = (x + residual) if residual is not None else x
        x = self.ln(residual)
        x = self.head(x)
        return x
