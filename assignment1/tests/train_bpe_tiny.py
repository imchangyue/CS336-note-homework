import json
import time
import os
import regex as re
from collections import Counter
import multiprocessing as mp
from typing import BinaryIO


# 将 find_chunk_boundaries 函数放在这里
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


# 新增一个用于每个进程的 worker 函数，它接收文件的起始和结束字节位置
def worker_tokenize_and_count(input_path: str, start: int, end: int, special_tokens: list[str]):
    """
    一个worker函数，用于处理文件的一个分块，进行预分词和频率统计。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        f.seek(start)
        data_chunk = f.read(end - start)

    delimiter_pattern = "|".join(re.escape(s) for s in special_tokens)
    parts = re.split(delimiter_pattern, data_chunk)
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    local_token_number = Counter()

    for part in parts:
        part_tokens = re.findall(PAT, part)
        for token in part_tokens:
            token_bytes = tuple(token.encode('utf-8'))
            local_token_number[token_bytes] += 1
            
    return local_token_number


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    print("开始并行预分词...")
    num_processes = mp.cpu_count()
    split_special_token_bytes = special_tokens[0].encode("utf-8") # 假设只有一个特殊token用于分割

    with open(input_path, 'rb') as f: # 注意这里是 "rb" 模式
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token_bytes)
    print("文本分割完成！")
    pool_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        pool_args.append((input_path, start, end, special_tokens))

    with mp.Pool(num_processes) as pool:
        # 使用 starmap 将参数元组传递给 worker 函数
        results = pool.starmap(worker_tokenize_and_count, pool_args)

    # 合并所有进程的结果
    token_number = Counter()
    for local_counter in results:
        token_number.update(local_counter)
    
    print("预分词完成，开始迭代合并...")
    
    # 初始化词汇表，并执行你的BPE迭代合并逻辑
    vocab = {i: bytes([i]) for i in range(256)}
    current_id = 256
    for token_str in special_tokens:
        vocab[current_id] = token_str.encode("utf-8")
        current_id += 1
    
    merges = []
    freq_dict = Counter()
    for token, count in token_number.items():
        number = len(token) - 1
        for i in range(number):
            pair = (token[i], token[i + 1])
            freq_dict[pair] += count
            
    while current_id < vocab_size:
        print(f"Current vocab size: {len(vocab)}, current_id: {current_id}, merges: {len(merges)}")
        if not freq_dict:
            break
        
        max_pair = max(freq_dict.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))
        merges.append((vocab[max_pair[0][0]], vocab[max_pair[0][1]]))
        
        freq_dict[max_pair[0]] = 0
        vocab[current_id] = vocab[max_pair[0][0]] + vocab[max_pair[0][1]]
        current_id += 1
        
        new_tokens = {}
        for token, count in token_number.items():
            updated_token = []
            i = 0
            number = len(token) - 1
            while i < number:
                pair = (token[i], token[i + 1])
                if pair == max_pair[0]:
                    updated_token.append(current_id - 1)
                    if i > 0:
                        freq_dict[(token[i - 1], token[i])] -= count
                        freq_dict[(token[i - 1], current_id - 1)] = freq_dict.get((token[i - 1], current_id - 1), 0) + count
                    if i + 1 < number:
                        freq_dict[(token[i + 1], token[i + 2])] -= count
                        freq_dict[(current_id - 1, token[i + 2])] = freq_dict.get((current_id - 1, token[i + 2]), 0) + count
                    i += 2
                else:
                    updated_token.append(token[i])
                    i += 1

            if i < len(token):
                updated_token.append(token[i])

            new_tokens[tuple(updated_token)] = count

        token_number = new_tokens

    return vocab, merges


# 你的主执行脚本
input_path = "/home/code/cs336/assignment1/data/owt_train.txt"
start_time = time.time()
vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=32000,
    special_tokens=["<|endoftext|>"]
)
end_time = time.time()
print(f"Total time: {end_time - start_time} seconds")

# 保存到文件
json_vocab = {str(k): v.decode('utf-8', errors='ignore') for k, v in vocab.items()}
json_merges = [[p1.decode('utf-8', errors='ignore'), p2.decode('utf-8', errors='ignore')] for p1, p2 in merges]

with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(json_vocab, f, ensure_ascii=False, indent=4)

with open('merges.json', 'w', encoding='utf-8') as f:
    json.dump(json_merges, f, ensure_ascii=False, indent=4)