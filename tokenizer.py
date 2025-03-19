import re
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch

KEEP_CHARS = set(
    [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "?",
        ".",
        ",",
        ":",
        "&",
        "%",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    ]
)


def split_len(splits):
    total = 0
    for s in splits:
        total += len(s)
    return total


def init_vocab(splits):
    vocab = set()
    for s in splits:
        for t in s:
            vocab.add(t)
    return list(vocab)


def compute_pair_frequencies(splits: list[list[str]]):
    pair_counts = Counter()
    for s in splits:
        pair_counts.update(zip(s, s[1:]))
    return pair_counts.most_common(1)[0]


def merge_pair_all(split, pair, pair_concat):
    result = []
    i = 0
    while i < len(split):
        if i < (len(split) - 1) and (pair[0] == split[i] and pair[1] == split[i + 1]):
            result.append(pair_concat)
            i += 1
        else:
            result.append(split[i])
        i += 1
    return result


def split_train(train):
    res = []
    for t in train:
        sub = []
        for c in t:
            sub.append(c)
        res.append(sub)
    return res


class BPE:
    def compute_bpe(self, train, n_vocab, vocab=None):
        splits = split_train(train)
        if vocab is None:
            vocab = init_vocab(splits)
        merges = []

        for _ in tqdm(range(n_vocab - len(vocab))):
            # find most frequent pair
            pair, freq = compute_pair_frequencies(splits)

            # add to history
            vocab_entry = pair[0] + pair[1]  # concat the string
            merges.append((pair, vocab_entry))
            vocab.append(vocab_entry)

            for i in range(len(splits)):
                splits[i] = merge_pair_all(splits[i], pair, vocab_entry)

        # save and return
        self.vocab = vocab
        self.merges = merges
        return self

    def assert_computed(self):
        assert self.vocab and self.merges, "First .compute_bpe() or load_bpe()"

    def encode(self, x: str):
        self.assert_computed()
        split = list(x)
        for m in self.merges:
            split = merge_pair_all(split, m[0], m[1])
        return split

    def save(self, filename: str):
        self.assert_computed()
        import pickle

        with open(filename, "wb") as f:
            pickle.dump((self.merges, self.vocab), f)

    def load(self, filename: str):
        import pickle

        with open(filename, "rb") as f:
            self.merges, self.vocab = pickle.load(f)
        return self


def vocab_to_idx_map(vocab, special_tokens=[]):
    t_to_i = {t: i for i, t in enumerate(special_tokens)}
    offset = len(special_tokens)
    for i, v in enumerate(vocab):
        t_to_i[v] = i + offset
    i_to_t = {t_to_i[t]: t for t in t_to_i}
    return t_to_i, i_to_t


def encode_idxs(bpe_split: list[str], t_to_i: dict[str, int]) -> list[int]:
    return [t_to_i[s] for s in bpe_split]


def normalize(x: str, keep_chars: set[str] = KEEP_CHARS):
    x = re.sub(r"\s", " ", x)  # make sure using the same space char (no special chars)
    x = re.sub(r"–", "-", x)  # use the - instead of –
    x = x.lower()  # only consider lower case chars

    # Keep chars we define and throw away others
    res = ""
    for c in x:
        if c in keep_chars:
            res += c
    return res


class Tokenizer:
    def __init__(self, bpe: BPE, special_tokens=[]):
        bpe.assert_computed()
        self.bpe = bpe
        self.special_tokens = special_tokens
        self.t_to_i, self.i_to_t = vocab_to_idx_map(bpe.vocab, special_tokens)

    @property
    def vocab_len(self):
        return len(self.special_tokens) + len(self.bpe.vocab)

    def encode(self, x: list[str]):
        result = []
        for s in x:
            if s in self.special_tokens:
                result.append(self.t_to_i[s])
            else:
                s = normalize(s)
                result.extend([self.t_to_i[s] for s in self.bpe.encode(s)])
        return result

    def decode(self, idxs: list[int]):
        bpe_x = [self.i_to_t[s] for s in idxs]
        return "".join(bpe_x)


class TitlesDataset:
    def __init__(self, df, tok_key="bpe_32k_2.11"):
        self.df: pd.DataFrame = df
        self.tok_key = tok_key

    def tokens_at_idx(self, i):
        return self.df.iloc[i][self.tok_key]

    def get_random_chunk(self, d_seq_len):
        SEP = 0
        # add separator at the beginning
        chunk = [SEP]
        idxs = set()
        total = 0
        # Get enough instances that we exceed d_seq_len
        while total < d_seq_len:
            # get random, but not seen yet
            i = np.random.randint(len(self.df))
            if i in idxs:
                continue
            idxs.add(i)

            # add instance with seperator
            r = self.tokens_at_idx(i)
            chunk.extend(r)
            chunk.append(SEP)
            total += len(r) + 1  # +1 since SEP appended
        return chunk

    def get_random_chunk_within_d_seq_len(self, d_seq_len):
        c = self.get_random_chunk(d_seq_len)

        target_size = d_seq_len + 1
        # perfectly sized!
        if len(c) == target_size:
            return c[:-1], c[1:]
        elif len(c) > target_size:
            # if too long, pick a chunk within
            i = np.random.randint(len(c) - d_seq_len)
            return c[i : i + d_seq_len], c[i + 1 : i + d_seq_len + 1]
        else:
            # Houston, we've got a problem
            raise Exception("Chunk should not be less than the target size")

    def get_random_batch(
        self,
        batch_size,
        d_seq_len,
    ):
        chunks = [self.get_random_chunk_within_d_seq_len(d_seq_len) for _ in range(batch_size)]
        return np.vstack([c[0] for c in chunks]), np.vstack([c[1] for c in chunks])

    def get_random_batch_tensor(self, batch_size, d_seq_len, device="cpu"):
        chunks = [self.get_random_chunk_within_d_seq_len(d_seq_len) for _ in range(batch_size)]
        return torch.tensor([c[0] for c in chunks], device=device), torch.tensor([c[1] for c in chunks], device=device)


if __name__ == "__main__":
    from datasets import load_dataset
    import numpy as np
    import os
    import pandas as pd

    print("Loading dataset")
    ds = load_dataset("laion/biorXiv_metadata")

    print("Converting to pandas and removing duplicates")
    df = ds["train"].to_pandas()
    df = df.drop_duplicates("doi")
    print(f"df loaded with {len(df)} entries")

    bpe_cache = "./data/bpe_32k_2.11.pickle"
    bpe = BPE()
    if os.path.exists(bpe_cache):
        bpe.load(bpe_cache)
        print("Loaded BPE from cache")
    else:
        np.random.seed(0)
        print("Computing BPE")
        train = df["title"].sample(32_000).parallel_apply(lambda x: normalize(x, KEEP_CHARS))
        bpe.compute_bpe(train=train.tolist(), n_vocab=2**11, vocab=list(KEEP_CHARS))
        bpe.save(bpe_cache)

    print("Precomputing Tokens for training later")
    t = Tokenizer(bpe, special_tokens=["<sep>"])

    df_cache = "./data/df_bpe_32k_2.11.parquet"
    if os.path.exists(df_cache):
        df = pd.read_parquet(df_cache)
    else:
        from pandarallel import pandarallel

        pandarallel.initialize(progress_bar=True)
        df["bpe_32k_2.11"] = df["title"].parallel_apply(lambda x: t.encode([x]))
        df = df[["doi", "bpe_32k_2.11"]]
        df.to_parquet(df_cache, index=False)

    print(df["bpe_32k_2.11"].head())
    print(len(df["bpe_32k_2.11"].array))
    ds = TitlesDataset(df, d_seq_len=64)
    bx, by = ds.get_random_batch(32)
    print(bx.shape, by.shape)
