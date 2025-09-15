import os
import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import List, Set, Tuple
import random
import time

def normalize_text(text: str) -> str:
    """
    标准化文本：小写、去标点、标准化空白、去重音、NFD Unicode标准化
    """
    # NFD Unicode标准化
    text = unicodedata.normalize('NFD', text)
    
    # 去重音符号
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # 转小写
    text = text.lower()
    
    # 去标点符号，只保留字母数字和空格
    text = re.sub(r'[^\w\s]', '', text)
    
    # 标准化空白（多个空格变成单个空格）
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def get_ngrams(text: str, n: int) -> Set[str]:
    """
    从文本中提取n-gram（以单词为单位）
    """
    words = text.split()
    if len(words) < n:
        return {' '.join(words)} if words else set()
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.add(ngram)
    
    return ngrams

def hash_function(text: str, seed: int) -> int:
    """
    使用种子创建哈希函数
    """
    hasher = hashlib.md5()
    hasher.update(f"{seed}_{text}".encode('utf-8'))
    return int(hasher.hexdigest(), 16)

def compute_minhash_signature(ngrams: Set[str], num_hashes: int) -> List[int]:
    """
    计算n-gram集合的MinHash签名
    """
    if not ngrams:
        return [0] * num_hashes
    
    signature = []
    
    for i in range(num_hashes):
        min_hash = min(hash_function(ngram, i) for ngram in ngrams)
        signature.append(min_hash)
    
    return signature

def lsh_bands(signature: List[int], num_bands: int) -> List[Tuple[int, ...]]:
    """
    将签名分成LSH带
    """
    band_size = len(signature) // num_bands
    bands = []
    
    for i in range(num_bands):
        start_idx = i * band_size
        end_idx = start_idx + band_size
        band = tuple(signature[start_idx:end_idx])
        bands.append(band)
    
    return bands

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    计算两个集合的Jaccard相似度
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    使用MinHash和LSH进行模糊文档去重
    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 存储每个文件的信息
    documents = {}  # file_path -> (normalized_text, ngrams, signature)
    
    # 第1步：读取文件并计算MinHash签名
    print("=" * 50)
    print("步骤1: 计算MinHash签名")
    print("=" * 50)
    
    for file_path in input_files:
        print(f"\n-> 正在处理文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 标准化文本
            normalized_text = normalize_text(content)
            print(f"   - 标准化后的文本: '{normalized_text[:50]}...'")
            
            # 提取n-grams
            doc_ngrams = get_ngrams(normalized_text, ngrams)
            print(f"   - 提取的 {ngrams}-grams 数量: {len(doc_ngrams)}")
            
            # 计算MinHash签名
            signature = compute_minhash_signature(doc_ngrams, num_hashes)
            print(f"   - 计算的 MinHash 签名: {signature[:10]}... (共 {len(signature)} 个值)")
            
            documents[file_path] = (normalized_text, doc_ngrams, signature)
            
        except Exception as e:
            print(f"   - 错误: 处理文件 {file_path} 时出错: {e}")
            continue
    
    # 第2步：使用LSH识别候选重复项
    print("\n" + "=" * 50)
    print("步骤2: 使用LSH识别候选重复项")
    print("=" * 50)
    
    band_buckets = defaultdict(set)  # band_hash -> set of file_paths
    
    for file_path, (_, _, signature) in documents.items():
        print(f"\n-> 正在为文件 {file_path} 生成 LSH 带...")
        bands = lsh_bands(signature, num_bands)
        print(f"   - 签名被分成 {len(bands)} 个带，每个带大小为 {len(signature) // num_bands}")
        
        for band_idx, band in enumerate(bands):
            # 为每个带创建唯一的桶标识符
            band_hash = hash(f"{band_idx}_{band}")
            band_buckets[band_hash].add(file_path)
            # print(f"   - 将文件 '{os.path.basename(file_path)}' 添加到带 {band_idx} 的桶 '{band_hash}'")
    
    # 收集候选重复项对
    candidate_pairs = set()
    print("\n-> 正在从 LSH 桶中收集候选对...")
    for bucket_hash, bucket_files in band_buckets.items():
        if len(bucket_files) > 1:
            print(f"   - 发现一个有 {len(bucket_files)} 个文件的桶: {bucket_files}")
            bucket_list = sorted(list(bucket_files))
            for i in range(len(bucket_list)):
                for j in range(i + 1, len(bucket_list)):
                    pair = tuple(sorted((bucket_list[i], bucket_list[j])))
                    candidate_pairs.add(pair)
    
    print(f"\n总计发现 {len(candidate_pairs)} 个候选重复项对")
    
    # 第3步：计算候选项的真实Jaccard相似度并识别重复项
    print("\n" + "=" * 50)
    print("步骤3: 计算真实Jaccard相似度")
    print("=" * 50)
    
    duplicate_pairs = []
    
    for file1, file2 in candidate_pairs:
        _, ngrams1, _ = documents[file1]
        _, ngrams2, _ = documents[file2]
        
        similarity = jaccard_similarity(ngrams1, ngrams2)
        print(f"-> 正在比较 '{os.path.basename(file1)}' 和 '{os.path.basename(file2)}'")
        print(f"   - 真实 Jaccard 相似度: {similarity:.4f}")
        
        if similarity >= jaccard_threshold:
            print(f"   - 相似度 {similarity:.4f} >= 阈值 {jaccard_threshold}。这是一个重复对！")
            duplicate_pairs.append((file1, file2, similarity))
    
    print(f"\n总计发现 {len(duplicate_pairs)} 个重复项对（阈值 >= {jaccard_threshold}）")
    
    # 第4步：聚类重复文档
    print("\n" + "=" * 50)
    print("步骤4: 聚类重复文档")
    print("=" * 50)
    
    # 使用并查集进行聚类
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 构建连通组件
    for file1, file2, _ in duplicate_pairs:
        union(file1, file2)
    
    # 收集聚类
    clusters = defaultdict(list)
    print("-> 正在构建聚类...")
    for file_path in documents.keys():
        root = find(file_path)
        clusters[root].append(file_path)
    
    print(f"   - 共形成 {len(clusters)} 个文档聚类")
    for i, cluster_files in enumerate(clusters.values()):
        print(f"     - 聚类 {i+1}: {cluster_files}")
    
    # 第5步：从每个聚类中随机选择一个文档保留
    print("\n" + "=" * 50)
    print("步骤5: 选择要保留的文档")
    print("=" * 50)
    
    files_to_keep = set()
    
    for cluster_files in clusters.values():
        if len(cluster_files) == 1:
            # 单个文件，直接保留
            files_to_keep.add(cluster_files[0])
            print(f"-> 单个文件，保留: {cluster_files[0]}")
        else:
            # 多个文件，随机选择一个保留
            selected_file = random.choice(cluster_files)
            files_to_keep.add(selected_file)
            print(f"-> 从聚类 {cluster_files} 中随机选择保留: {selected_file}")
    
    # 第6步：将保留的文件写入输出目录
    print("\n" + "=" * 50)
    print("步骤6: 写入输出文件")
    print("=" * 50)
    
    for file_path in input_files:
        if file_path in files_to_keep:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_directory, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as input_file:
                    content = input_file.read()
                
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(content)
                print(f"-> 文件已复制到: {output_path}")
                
            except Exception as e:
                print(f"   - 错误: 写入文件 {output_path} 时出错: {e}")
    
    print("\n" + "=" * 50)
    print("去重完成！")
    print(f"保留了 {len(files_to_keep)} 个文件，共 {len(input_files)} 个输入文件")
    print(f"删除了 {len(input_files) - len(files_to_keep)} 个重复文件")
    print("=" * 50)


# 使用示例（你需要创建示例文件来运行此代码）
if __name__ == "__main__":
    
    # 创建一些示例文件
    if not os.path.exists("test_deduplication"):
        os.makedirs("test_deduplication")
    
    with open("test_deduplication/file1.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample document for testing the MinHash deduplication algorithm. It has some unique content.")
    
    with open("test_deduplication/file2.txt", "w", encoding="utf-8") as f:
        f.write("This document is a sample for testing the MinHash deduplication. It is very similar but not identical.")

    with open("test_deduplication/file3.txt", "w", encoding="utf-8") as f:
        f.write("A third document, completely different from the others, to ensure unique files are handled correctly.")
    
    with open("test_deduplication/file4.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample document for testing the minhash deduplication algorithm. It has some unique content.")

    input_files = [
        "test_deduplication/file1.txt",
        "test_deduplication/file2.txt",
        "test_deduplication/file3.txt",
        "test_deduplication/file4.txt"
    ]
    
    run_minhash_deduplication(
        input_files=input_files,
        num_hashes=100,
        num_bands=20,
        ngrams=3,
        jaccard_threshold=0.8,
        output_directory="deduplicated_output"
    )