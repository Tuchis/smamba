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
                 device: torch.device = torch.device('cpu'),
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_expanded = d_expanded
        self.d_rank = d_rank
        self.d_hidden = d_hidden
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.device = device
        self.cached_state = None

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
        self.A_log = nn.Parameter(torch.log(A)).to(device)
        self.A_log._no_weight_decay = True

    def forward(self, x: torch.Tensor, cache: torch.Tensor = False, one_step: torch.Tensor = False):
        """
        x: torch.Tensor (B, L, d_expanded)
        """
        if one_step:
            assert x.shape[1] == 1, "one_step mode only supports L=1"
        if one_step and self.cached_state is None:
            raise ValueError("Cache is empty. Please run the model in non-one_step mode first.")
        dt, B, C = torch.split(self.x_proj(x), [self.d_rank, self.d_hidden, self.d_hidden], dim=-1)
        A = -torch.exp(self.A_log) # (d_expanded, d_hidden)
        dt = F.linear(dt, self.dt_proj.weight)  # (B L d_expanded) or (B d_expanded)
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype)) # (B L d_expanded) or (B d_expanded)
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A)) # (B L d_expanded d_hidden)
        dB = torch.einsum("bld,bln->bldn", dt, B) # (B L d_expanded d_hidden)
        dBX = torch.einsum("bldn,bld->bldn", dB, x) # (B L d_expanded d_hidden)
        if one_step:
            hidden_state = self.cached_state[:, -1:]*dA + dBX
            self.cached_state = torch.cat([self.cached_state, hidden_state], dim=1)
            return torch.einsum("bldn,bln->bld", hidden_state, C)
        
        for i in range(int(math.log2(x.shape[1]))):
            dBX_copy = dBX.clone()
            dBX_copy[:, 2**i:] = dA[:, 2**i:]*dBX[:, :-2**i] + dBX[:, 2**i:]
            dA_copy = dA.clone()
            dA_copy[:, 2**i:] = dA[:, 2**i:]*dA[:, :-2**i]
            dA = dA_copy
            dBX = dBX_copy
        
        if cache:
            self.cached_state = dBX

        y = torch.einsum("bldn,bln->bld", dBX_copy, C)
        return y

class MambaBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 expansion: int = 2,
                 d_rank: int = None,
                 d_hidden: int = 16,
                 dt_min: float = 1e-3,
                 dt_max: float = 1e-1,
                 kernel_size: int = 5,
                 device: torch.device = torch.device('cpu'),
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
        self.conv_cache = None

        self.in_proj = nn.Linear(d_model, 2*self.d_expanded, device=device)
        self.out_proj = nn.Linear(self.d_expanded, d_model, device=device)

        self.conv1d = nn.Conv1d(self.d_expanded, self.d_expanded, self.kernel_size, bias=True, padding=(self.kernel_size-1)//2, groups=self.d_expanded, device=device)
        self.ssm = SSM(self.d_expanded, self.d_rank, self.d_hidden, self.dt_min, self.dt_max, self.device)

    def forward(self, x: torch.Tensor, cache: torch.Tensor = False, one_step: torch.Tensor = False):
        """
        x: torch.Tensor (B, L, d_model)
        """
        if one_step and self.conv_cache is None:
            raise ValueError("Cache is empty. Please run the model in non-one_step mode first.")
        x, z = torch.split(self.in_proj(x), [self.d_expanded, self.d_expanded], dim=-1)
        if cache:
            unpadded = self.kernel_size - (self.kernel_size-1)//2 - 1
            self.conv_cache = x[:, -unpadded:]
        if one_step:
            x = torch.cat([self.conv_cache, x], dim=1)
            self.conv_cache = x[:, 1:]
            x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
            x = x[:, -1:]
        else:
            x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.ssm(x, cache=cache, one_step=one_step)
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
                    kernel_size: int = 5,
                    device: torch.device = torch.device('cpu'),
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

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None, cache: torch.Tensor = False, one_step: torch.Tensor = False):
        """
        x: torch.Tensor (B, L, d_model)
        residual: torch.Tensor (B, L, d_model)
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.ln(hidden_states)
        hidden_states = self.mamba(hidden_states, cache=cache, one_step=one_step)
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
                 kernel_size: int = 5,
                 device: torch.device = torch.device('cpu'),
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

    def forward(self, x: torch.Tensor, cache: torch.Tensor = False, one_step: torch.Tensor = False):
        """
        x: torch.Tensor (B, L)
        """
        if one_step:
            return self.step(x)
        residual = None
        x = self.token_emb(x) # (B, L, d_model)
        for block in self.blocks:
            x, residual = block(x, cache=cache) # (B, L, d_model)
        residual = (x + residual) if residual is not None else x
        x = self.ln(residual) # (B, L, d_model)
        x = x[:, -1] # (B, d_model)
        x = self.head(x) # (B, num_tokens)
        return x

    def step(self, x: torch.Tensor):
        """
        x: torch.Tensor (B)
        """
        residual = None
        x = self.token_emb(x) # (B, d_model)
        for block in self.blocks:
            x, residual = block(x, one_step=True) # (B, d_model)
        residual = (x + residual) if residual is not None else x
        x = self.ln(residual) # (B, d_model)
        x = self.head(x) # (B, num_tokens)
        return x
