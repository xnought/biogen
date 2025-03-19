from model import Transformer
from tokenizer import TitlesDataset, BPE, Tokenizer
import pandas as pd
import torch
import numpy as np
import os


def get_random_batch_tensor(ds, batch_size, device):
    x, y = ds.get_random_batch(batch_size)
    X = torch.tensor(x, device=device)
    Y = torch.tensor(y, device=device)
    return X, Y


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BPE_PATH = os.path.join("data", "bpe_32k_2.11.pickle")
    DATASET_PATH = os.path.join("data", "df_bpe_32k_2.11.parquet")

    tokenizer = Tokenizer(BPE().load(BPE_PATH), special_tokens=["<sep>"])
    d_seq_len = 32
    d_in = 64
    d_k = 64
    d_out = 64
    n_heads = 2
    n_blocks = 2
    n_vocab = len(tokenizer.bpe.vocab) + len(tokenizer.special_tokens)

    df = pd.read_parquet(DATASET_PATH)
    ds = TitlesDataset(df, d_seq_len)

    model = Transformer(n_vocab=n_vocab, n_heads=n_heads, d_seq_len=d_seq_len, d_in=d_in, d_k=d_k, d_out=d_out, n_blocks=n_blocks)
    model = model.to(DEVICE)
    print(model)

    batch_size = 32
    X, Y = get_random_batch_tensor(ds, batch_size, DEVICE)
    loss, _ = model.train_step(X, Y)
    print(loss)
