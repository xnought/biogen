from model import TransformerBlock
from tokenizer import TitlesDataset
import pandas as pd
import torch
import numpy as np


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    B = 2
    d_seq_len = 8
    d_in = 3
    d_k = 3
    d_out = 3

    df = pd.read_parquet("./data/df_bpe_32k_2.11.parquet")
    ds = TitlesDataset(df, d_seq_len)

    X = torch.randn((B, d_seq_len, d_in))
    print(X)

    n_heads = 4
    b = TransformerBlock(n_heads, d_seq_len, d_in, d_k, d_out)
    print(b)
    print(b(X))
