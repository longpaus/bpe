from __future__ import annotations

from typing import Optional
import regex as re
from pycparser.ply.yacc import token

from src.utils.tokenizer import merge


def _get_key(d:dict, target_value):
    return next((key for key, value in d.items() if value == target_value), None)


class BPETokenizer:
    def __init__(self, vocab:dict[int, bytes], merges:list[tuple[bytes,bytes]], special_tokens:Optional[list[str]]=None):
        self.vocab = vocab
        self.merges = merges

        # add escape lines to special tokens as they may contain '|'
        self.special_tokens = special_tokens
    def encode(self, text:str) -> list[int]:
        pre_tokens: list[bytes|str] = self._pre_tokenize(text)
        token_ids = []
        for pre_token in pre_tokens:
            if type(pre_token) == bytes:
                t = tuple(bytes([byte]) for byte in pre_token)
                for m in self.merges:
                    optional_t = merge(t, m)
                    t = optional_t if optional_t else t
                t_ids = list(map(lambda x:_get_key(self.vocab, x), t))
                token_ids.extend(t_ids)
            elif pre_token in self.special_tokens:
                token_ids.append(_get_key(self.vocab, pre_token))
        return token_ids

    def decode(self, tokens:list[int]) -> str:
        texts = []
        for tok in tokens:
            text = self.vocab[tok]
            if type(text) == bytes:
                text = text.decode()
            texts.append(text)
        return "".join(texts)
    def _pre_tokenize(self, chunk:str)-> list[bytes|str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        split_chunk:list[str] = [chunk]
        # remove special tokens
        special_tokens_found:Optional[list[str]] = None
        if self.special_tokens:
            pattern = "|".join(list(map(lambda st: re.escape(st), self.special_tokens)))
            split_chunk:list[str] = re.split(pattern, chunk)
            special_tokens_found = re.findall(pattern, chunk)
        pre_tokens:list[bytes|str] = [] # either contain bytes or str (special tokens)

        for i,sub_chunk in enumerate(split_chunk):
            pre_tok:list[bytes] = [m.group(0).encode('utf-8') for m in re.finditer(PAT, sub_chunk)]
            pre_tokens.extend(pre_tok)
            if special_tokens_found and i < len(split_chunk)-1:
                pre_tokens.append(special_tokens_found[i])
        return pre_tokens
