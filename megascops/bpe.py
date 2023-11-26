import re
import collections


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def learn_bpe(vocab: dict, num_merges: int = 10):
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)


def main():
    # https://www.youtube.com/watch?v=HEikzVL-lZU
    my_vocab = {
        "l o w <eos>": 5,
        "l o w e r <eos>": 2,
        "n e w e s t <eos>": 6,
        "w i d e s t <eos>": 3,
    }
    learn_bpe(my_vocab, 15)