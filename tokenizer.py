import re
from tqdm import tqdm
from collections import Counter

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


if __name__ == "__main__":
    from datasets import load_dataset
    import numpy as np
    import os

    print("Loading dataset")
    ds = load_dataset("laion/biorXiv_metadata")

    print("Converting to pandas and removing duplicates")
    df = ds["train"].to_pandas()
    df = df.drop_duplicates("doi")
    print(f"df loaded with {len(df)} entries")

    cache = "./data/bpe_32k_2.11.pickle"
    bpe = BPE()
    if os.path.exists(cache):
        bpe.load(cache)
        print("Loaded BPE from cache")
    else:
        np.random.seed(0)
        print("Computing BPE")
        bpe.compute_bpe(train=df["parsed_title"].sample(32_000).tolist(), n_vocab=2**11, vocab=list(KEEP_CHARS))
        bpe.save(cache)

    t = Tokenizer(bpe, ["<s>"])
    e = t.encode(["<s>", "Hello World!", "<s>"])
    print(e)
    out = t.decode(e)
    print(out)
