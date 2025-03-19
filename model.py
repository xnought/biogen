import torch


def attn(Q, K, V, mask):
    QKT = (Q @ K.transpose(-2, -1)) * (K.shape[-1] ** -0.5)
    masked_QKT = QKT.masked_fill(mask, float("-inf"))
    weights = torch.softmax(masked_QKT, dim=-1)
    return weights @ V


def uniform_parameter(size, a, b, requires_grad=True):
    t = torch.empty(size, requires_grad=requires_grad)
    torch.nn.init.uniform_(t, a=a, b=b)
    return torch.nn.Parameter(t)


class MHLinear(torch.nn.Module):
    def __init__(self, n_heads, d_in, d_k):
        """Linear transformation (no Bias) but applied to multiple n_heads hence MHLinaer"""
        super().__init__()
        b = d_in**-0.5
        a = -b
        self.weights = uniform_parameter((d_in, n_heads, d_k), a, b)

    def __call__(self, X):
        B, d_seq_len, d_in = X.shape
        _, n_heads, d_k = self.weights.shape

        X_unrolled = X.view(-1, d_in)  # (B*d_seq_len, d_in)
        W_unrolled = self.weights.view(d_in, -1)  # (d_in, n_heads*d_k)
        projected = X_unrolled @ W_unrolled  # (B*d_seq_len, n_heads*d_k)

        return projected.view((B, d_seq_len, n_heads, d_k)).transpose(1, 2).contiguous()  # (B, n_heads, d_seq_len, d_k)


class MHA(torch.nn.Module):
    def __init__(self, n_heads, d_seq_len, d_in, d_k, d_out):
        super().__init__()
        self.Q = MHLinear(n_heads, d_in, d_k)
        self.K = MHLinear(n_heads, d_in, d_k)
        self.V = MHLinear(n_heads, d_in, d_k)
        self.causal_mask = torch.triu(
            torch.ones((d_seq_len, d_seq_len), requires_grad=False, dtype=torch.bool),
            diagonal=1,
        )
        self.out = torch.nn.Linear(n_heads * d_k, d_out, bias=False)

    def __call__(self, X):
        B, d_seq_len, d_in = X.shape
        mha = attn(Q=self.Q(X), K=self.K(X), V=self.V(X), mask=self.causal_mask)  # (B, n_heads, d_seq_len, d_k)
        mha = mha.transpose(1, 2).contiguous().view((B, d_seq_len, -1))  # (B, d_seq_len, n_heads*d_k)
        return self.out(mha)  # project to d_out


def gen_mlp(n_layers, d_out):
    modules = []
    for _ in range(n_layers):
        modules.append(torch.nn.Linear(d_out, d_out))
        modules.append(torch.nn.ReLU())
    return torch.nn.Sequential(*modules)


class TransformerBlock(torch.nn.Module):
    def __init__(self, n_heads, d_seq_len, d_in, d_k, d_out, n_layers=2):
        super().__init__()
        self.mha = MHA(n_heads, d_seq_len, d_in, d_k, d_out)
        self.mha_norm = torch.nn.LayerNorm((d_seq_len, d_out))

        self.mlp = gen_mlp(n_layers, d_out)
        self.mlp_norm = torch.nn.LayerNorm((d_seq_len, d_out))

    def __call__(self, X):
        X = self.mha_norm(X + self.mha(X))
        X = self.mlp_norm(X + self.mlp(X))
        return X


class Transformer(torch.nn.Module):
    def __init__(self, n_heads, d_seq_len, d_in, d_k, d_out, n_layers=2, n_blocks=5):
        super().__init__()
        pass
        # self.blocks = [TransformerBlock(n_heads, d_seq_len, d_in, d_k, d_out, n_layers) for _ in n_blocks]
