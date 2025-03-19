from model import Transformer
from tokenizer import TitlesDataset, BPE, Tokenizer
import pandas as pd
import torch
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BPE_PATH = os.path.join("data", "bpe_32k_2.11.pickle")
DATASET_PATH = os.path.join("data", "df_bpe_32k_2.11.parquet")


def get_random_batch_tensor(ds, batch_size, device):
    x, y = ds.get_random_batch(batch_size)
    X = torch.tensor(x, device=device)
    Y = torch.tensor(y, device=device)
    return X, Y


def df_split(df, percent_train=0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = int(percent_train * len(df))
    return df.iloc[:split_at], df.iloc[split_at:]


@torch.no_grad()
def test_performance(model: Transformer, test: TitlesDataset, iters: int, batch_size: int, d_seq_len: int, device: torch.device):
    model.eval()
    avg_loss = 0
    for i in range(iters):
        X, Y = test.get_random_batch_tensor(batch_size, d_seq_len, device)
        loss, _ = model.train_step(X, Y)
        avg_loss += loss.cpu().item()
    model.train()
    return avg_loss / iters


def save_model(model: torch.nn.Module, filepath: str):
    torch.save(model.state_dict(), filepath)


def train_model(
    model: Transformer,
    optim: torch.optim.Optimizer,
    train: TitlesDataset,
    test: TitlesDataset,
    iters: int = 1_000,
    test_iters: int = 3,
    batch_size: int = 64,
    d_seq_len: int = 32,
    device: torch.device = DEVICE,
    checkpoints_every=100,
):
    perf_history = {"train_loss": [-1.0] * iters, "test_loss": [-1.0] * iters}

    model.train()
    for i in range(iters):
        # forward pass
        X, Y = train.get_random_batch_tensor(batch_size, d_seq_len, device)
        loss, _ = model.train_step(X, Y)

        # backward pass and update
        optim.zero_grad()
        loss.backward()
        optim.step()

        # log and save performance
        test_loss = test_performance(model, test, test_iters, batch_size, d_seq_len, device)
        train_loss = loss.detach().cpu().item()
        perf_history["train_loss"][i] = train_loss
        perf_history["test_loss"][i] = test_loss
        print(f"ITER: {i + 1}/{iters}\tTRAIN: {train_loss}\tTEST: {test_loss}")

        if i % checkpoints_every == 0:
            save_model(model, "checkpoint.pth")

    return perf_history


def plot_performance(perf_history: dict[str, list[float]]):
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    plot = lambda key: plt.plot(perf_history[key], label=key)
    plot("train_loss")
    plot("test_loss")
    plt.ylabel("Loss")
    plt.xlabel("Train Iteration")
    plt.legend()
    plt.show()


def load_datasets():
    train_df, test_df = df_split(df=pd.read_parquet(DATASET_PATH), percent_train=0.95)
    train = TitlesDataset(train_df)
    test = TitlesDataset(test_df)
    return train, test


def generate(string, model, tokenizer, num_gen=None, device="cpu", print=lambda x: x):
    genned = 0
    while True:
        tokens = torch.tensor(tokenizer.encode(["<sep>", string]), device=device)
        pred = model.generate(tokens, device=device)
        genned += 1
        if num_gen is None and pred == 0:
            break
        elif num_gen is not None and genned > num_gen:
            break

        string += tokenizer.decode([pred])
        print(string)

    return string


def test_gen():
    tokenizer = Tokenizer(BPE().load(BPE_PATH), special_tokens=["<sep>"])
    model_config = dict(
        d_seq_len=32,
        d_in=64,
        d_k=64,
        d_out=64,
        n_heads=2,
        n_blocks=2,
        n_vocab=tokenizer.vocab_len,
    )
    model = torch.compile(Transformer(**model_config)).to(DEVICE)
    model.load_state_dict(torch.load("checkpoint.pth"))
    print(generate("evidence for ", num_gen=100, model=model, tokenizer=tokenizer, device=DEVICE))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    print("COMPILING MODEL")
    tokenizer = Tokenizer(BPE().load(BPE_PATH), special_tokens=["<sep>"])
    model_config = dict(
        d_seq_len=32,
        d_in=64,
        d_k=64,
        d_out=64,
        n_heads=2,
        n_blocks=2,
        n_vocab=tokenizer.vocab_len,
    )
    model = torch.compile(Transformer(**model_config)).to(DEVICE)

    train_config = dict(iters=100, test_iters=3, batch_size=32, lr=1e-3, d_seq_len=model_config["d_seq_len"])
    train, test = load_datasets()

    print("TRAINING FOR ", train_config)
    optim = torch.optim.AdamW(model.parameters(), lr=train_config["lr"])
    perf = train_model(
        model=model,
        optim=optim,
        train=train,
        test=test,
        iters=train_config["iters"],
        test_iters=train_config["test_iters"],
        batch_size=train_config["batch_size"],
        d_seq_len=train_config["d_seq_len"],
        device=DEVICE,
        checkpoints_every=20,
    )
    plot_performance(perf)
