from typing import List, Tuple, Optional, Any, Iterator,Iterable, Iterator, Dict
import os
import regex as re
from array import array
import heapq
from collections import defaultdict, Counter
from functools import total_ordering
import json

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[bytes]:
    """使用GPT-2的正则表达式将文本分割成“词块”，并编码为bytes。 This step is very important!!!! Otherwise the b'a\n\nb' will be transfer into 'a' '\n\n' 'b' instead of 'a' '\n' '\n' 'b'"""
    str_tokens = re.findall(GPT2_SPLIT_PATTERN, text)
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    return byte_tokens


GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)
def iter_pretokenize(text: str) -> Iterator[bytes]:
    """按 GPT-2 正则逐个产生字节串，零内存列表。"""
    for m in GPT2_RE.finditer(text):
        yield m.group(0).encode("utf-8")
class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        # Initialize vocabulary, merges, and special tokens
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        self.stoi = {v: k for k, v in vocab.items()}
        self.itos = vocab
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in merges}
        # Ensure special tokens are added to the vocab
        for token in self.special_tokens:
            token_bytes = token.encode()
            if token_bytes not in self.vocab.values():
                new_id = max(self.vocab.keys()) + 1  # Create new ID
                self.vocab[new_id] = token_bytes

        # Preprocess merges into a dictionary for quick lookup
        self.merge_dict = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_bytes = {token: id for id, token in self.vocab.items()}
    def _encode_ordinary_text(self, text_bytes: bytes) -> list[int]:
        if not text_bytes:
            return []

        try:
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = text_bytes.decode("utf-8", errors="replace")

        ids_out = array("H")  # uint16 足够 ≤ 65k vocab
        pair_rank = self.merges_rank
        pair2new = self.pair2new
        byte2id = self.stoi  # 局部 alias，加速

        # 逐个“词块”处理，避免一次性 list
        for word_b in iter_pretokenize(text):
            token_ids = array("H", (byte2id[bytes([b])] for b in word_b))

            # b. 就地合并：“greedy smallest-rank merge”
            while True:
                best_rank = 1000000000
                best_pos = -1
                # ——— 找当前序列里 rank 最小的 pair ———
                for i in range(len(token_ids) - 1):
                    r = pair_rank.get( # ——— 替换 best_pos & best_pos+1 为新的 token ———
                        (self.itos[token_ids[i]], self.itos[token_ids[i + 1]]),
                        1000000000,
                    )
                    if r < best_rank:
                        best_rank, best_pos = r, i
                if best_pos == -1:
                    break
                
                new_id = pair2new[
                    (self.itos[token_ids[best_pos]], self.itos[token_ids[best_pos + 1]])
                ]
                token_ids[best_pos : best_pos + 2] = array("H", [new_id])

            ids_out.extend(token_ids)

        # array → list
        return ids_out.tolist()
    def encode(self, text: str) -> List[int]:
        if text == '':
            return [] 
        """Encode a string into a list of token IDs using BPE"""
        # Pre-tokenization: Split by spaces
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

            # 构建 delimiter_pattern
            delimiter_pattern = r"(" + r"|".join([re.escape(token) for token in sorted_special_tokens]) + r")"

            # 按照 delimiter_pattern 进行分割
            parts = re.split(delimiter_pattern, text)
        else:
            parts = [text]

        token_ids = []
        for part in parts:
            if part in self.special_tokens:
                # 直接编码 special token
                token_ids.append(self.get_token_id_from_bytes(part.encode('utf-8')))
            else:
                token_ids.extend(self._encode_ordinary_text(part.encode("utf-8")))
                
        return token_ids


#"hello" -> "h":以h开头的100个bytes都相符->错
#                以h开头的99个bytes都相符->错
#              # 以h开头的98个bytes都相符->对
#  这个方法是错的
#错在哪里："hello" ->"he" 和 "llo" 不能继续合并成hello
#         "hello" ->"h" 和 "ello"可以继续合并成hello

# 从merge的顺序进行encode 对:he 和 el谁先合并的问题
# 先从原来训练的顺序开始合并，自然可以合并成一样
    
    def decode(self, ids: List[int]) -> str:
        """解码函数，将 token IDs 转换回文本"""
        tokens = [self.vocab.get(id, b"") for id in ids]
        # 将字节元组连接成单个bytes对象
        combined_bytes = b''.join(tokens)

        # 使用UTF-8解码
        result = combined_bytes.decode('utf-8', errors='ignore')
        return result
    
    def tokenize(self, text: str) -> List[str]:
        """将输入的文本拆分成tokens"""
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = re.findall(PAT, text)  # 按空格分词，同时保留空格
        return tokens
    
    def get_token_id_from_bytes(self, token_bytes: bytes) -> int:
        """从 token 字节获取其对应的 token ID"""
        for token_id, token in self.vocab.items():
            if token == token_bytes:
                return token_id
        return -1  # 如果找不到对应的 token，返回一个默认值（例如 -1）

    def apply_merges(self, tokens: List[str]) -> List[str]:
        """根据合并规则对 tokens 进行合并"""
        pairs = self.get_pairs(tokens)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.merge_dict.get(pair, float('inf')))
            if bigram not in self.merge_dict:
                break
            first, second = bigram
            new_token = first + second
            tokens = self.merge(tokens, first, second, new_token)
            pairs = self.get_pairs(tokens)
        return tokens
    
    def get_pairs(self, tokens: List[str]) -> set:
        """获取当前 tokens 中所有相邻的 token 对"""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def merge(self, tokens: List[str], first: str, second: str, new_token: str) -> List[str]:
        """将 token 对（first, second）合并为 new_token"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == first and i + 1 < len(tokens) and tokens[i + 1] == second:
                new_tokens.append(new_token)
                i += 2  # 跳过已经合并的部分
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性编码函数，用于处理大文件"""
        for line in iterable:
            yield from self.encode(line)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        """从文件中加载 vocab 和 merges，构建 Tokenizer 实例"""
        # 加载 vocab
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                token, id = line.strip().split()
                vocab[int(id)] = token.encode()

        # 加载 merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                token1, token2 = line.strip().split()
                merges.append((token1.encode(), token2.encode()))

        return cls(vocab, merges, special_tokens)
