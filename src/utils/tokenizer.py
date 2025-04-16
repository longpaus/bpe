from __future__ import annotations

import os
from typing import BinaryIO
import regex as re

def get_adj_pairs(sequence:tuple[bytes,...]) -> list[tuple[bytes, bytes]]:
    return [(a,b) for a, b in zip(sequence, sequence[1:])]

def get_most_freq_pair(pair_count:dict[tuple[bytes,bytes], int]) -> tuple[bytes,bytes]:
    max_val = max(pair_count.values())
    return max([key for key, val in pair_count.items() if val == max_val])

def combine_pair(pair:tuple[bytes,bytes])-> bytes:
    return b''.join(pair)

def remove_indices(lst:list, indices:list[int]):
    indices_set = set(indices)
    return [lst[i] for i in range(len(lst)) if i not in indices_set]

def merge(word:tuple[bytes,...], pair:tuple[bytes,bytes])-> tuple[bytes,...]|None:
    """
    Merges consecutive occurrences of a specified pair of bytes in a word into a single combined byte.

    Args:
        word (tuple[bytes, ...]): A tuple of bytes representing the input sequence.
        pair (tuple[bytes, bytes]): A tuple of two bytes to be merged when found consecutively.

    Returns:
        tuple[bytes, ...] | None: A tuple with all pairs merged, or None if no pairs were found.
    """
    if len(pair) > len(word):
        return None

    l1, l2 = pair
    new = list(word)
    i = 0
    pair_found = False

    while i < len(new) - 1:
        if new[i] == l1 and new[i + 1] == l2:
            new[i] = combine_pair(pair)
            new.pop(i + 1)  # Remove the second byte of the pair
            pair_found = True
            i += 1  # Skip past the merged byte
        else:
            i += 1

    return tuple(new) if pair_found else None

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(chunk:str, special_tokens:list[str]) -> dict[tuple[bytes, ...], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    split_data = [chunk]
    # remove special tokens
    if special_tokens:
        split_data:list[str] = re.split("|".join(special_tokens), chunk)

    # pre-tokenize
    freq_tab:dict[tuple[bytes,...],int] = {} # each key is a tuple of bytes where each bytes are a vocab
    for d in split_data:
        pre_tok:list[bytes] = [m.group(0).encode('utf-8') for m in re.finditer(PAT, d)]
        for w in pre_tok:
            key = tuple(bytes([byte]) for byte in w)
            if key in freq_tab:
                freq_tab[key] += 1
            else:
                freq_tab[key] = 1
    return freq_tab