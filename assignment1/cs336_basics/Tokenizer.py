from typing import List, Tuple, Optional, Any, Iterator
from collections.abc import Iterable
# import re
import regex as re
class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        # 初始化词汇表、合并规则和特殊 tokens
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 如果有特殊token，确保它们被加到词汇表中
        for token in self.special_tokens:
            token_bytes = token.encode()
            if token_bytes not in self.vocab.values():
                new_id = max(self.vocab.keys()) + 1  # 创建新的 ID
                self.vocab[new_id] = token_bytes
        
        # 预处理合并规则
        self.merge_dict = {pair: i for i, pair in enumerate(self.merges)}


    def encode(self, text: str) -> List[int]:
        """编码函数，将文本转换为 token IDs，支持多字节 token 和 special tokens"""
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

            # 构建 delimiter_pattern
            delimiter_pattern = r"(" + r"|".join([re.escape(token) for token in sorted_special_tokens]) + r")"

            # 按照 delimiter_pattern 进行分割
            parts = re.split(delimiter_pattern, text)
        else:
            parts = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_ids = []
        max_token_length = max(len(token) for key,token in self.vocab.items())
        for part in parts:
            if part in self.special_tokens:
                # 直接编码 special token
                token_ids.append(self.get_token_id_from_bytes(part.encode('utf-8')))
            else:
                sub_tokens = re.findall(PAT, part)
                for token in sub_tokens:
                    encoded_bytes = token.encode('utf-8')
                    start = 0
                    while start < len(encoded_bytes):
                        max_possible_length = min(max_token_length, len(encoded_bytes) - start)
                        found = False
                        for length in range(max_possible_length, 0, -1):
                            candidate_bytes = encoded_bytes[start:start + length]
                            if candidate_bytes in self.vocab.values():
                                token_ids.append(self.get_token_id_from_bytes(candidate_bytes))
                                start += length
                                found = True
                                break
                        if not found:
                            token_ids.append(self.get_token_id_from_bytes(b""))  # 或使用 UNK
                            start += 1

        return token_ids

    
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
