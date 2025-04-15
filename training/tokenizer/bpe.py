from __future__ import annotations

import os
import regex as re
import tokenizer.utils as bpe_utils
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab:dict[int, bytes] = {}
    merges:list[tuple[bytes,bytes]] = []

    # add escape lines to special tokens as they may contain '|'
    special_tokens = list(map(lambda st: re.escape(st), special_tokens))

    # add special tokens to vocab
    for i,special_token in enumerate(special_tokens):
        vocab[i] = special_token

    num_processes = 8

    chunks = []
    # get chunks of text from file path
    with open(input_path, "rb") as f:
        boundaries = bpe_utils.find_chunk_boundaries(
            f,num_processes, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
    # perform pre-tokenization on the chunks with multiple processes
    with Pool(processes=num_processes) as pool:
        pre_tok_func = partial(bpe_utils.pre_tokenize, special_tokens=special_tokens)
        process_tabs:list[dict[tuple[bytes,...],int]] = pool.map(pre_tok_func, chunks)
    freq_tab: dict[tuple[bytes, ...], int]  = {}
    # aggregate results
    for tab in process_tabs:
        for k, v in tab.items():
            freq_tab[k] = freq_tab.get(k,0) + v

    for i in tqdm(range(len(special_tokens),vocab_size)):
        pairs_count:dict[tuple[bytes,bytes], int] = {}
        for byte_seq in freq_tab:
            count = freq_tab[byte_seq]
            adj_pairs = bpe_utils.get_adj_pairs(byte_seq)

            # count pairs
            for adj_pair in adj_pairs:
                if adj_pair in pairs_count:
                    pairs_count[adj_pair] += count
                else:
                    pairs_count[adj_pair] = count
        most_freq_pair = bpe_utils.get_most_freq_pair(pairs_count)
        merges.append(most_freq_pair)
        vocab[i] = bpe_utils.combine_pair(most_freq_pair)

        # merging bytes in the frequency table
        for w in list(freq_tab.keys()):
            merged_word = bpe_utils.merge(w, most_freq_pair)
            if merged_word:
                freq_tab[merged_word] = freq_tab.pop(w)
    return vocab, merges