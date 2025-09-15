from __future__ import annotations
import os
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re
import hashlib
import unicodedata
from collections import defaultdict
from typing import List, Set, Tuple
import random

# 在全局作用域加载模型以提高效率，避免重复加载
try:
    model = fasttext.load_model('/data/classifiers/lid.176.bin')
    print("fastText model loaded successfully.")
except ValueError as e:
    # 如果模型加载失败，打印错误并让 model 变量为 None
    print(f"Error loading fastText model: {e}")
    model = None

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        # 1. 自动检测编码并解码。
        # 修复：resiliparse 的 detect_encoding() 函数在某些版本中可能只返回一个字符串，而不是一个元组。
        # 这里的代码假设它直接返回编码名称。
        encoding = detect_encoding(html_bytes)
        html_string = html_bytes.decode(encoding, errors='replace')
        
        # 2. 从解码后的HTML字符串中提取纯文本。
        plain_text = extract_plain_text(html_string)
        
        return plain_text
        
    except Exception as e:
        # 如果在解码或提取过程中发生任何错误，返回 None。
        print(f"处理HTML时出错: {e}")
        return None




def run_identify_language(text: str) -> tuple[str, float]:
    """
    使用 fastText 语言识别模型识别输入字符串的语言和置信度。
    
    参数:
    text: 要识别的 Unicode 字符串。
    
    返回:
    一个包含 (language_identifier, confidence_score) 的元组。
    """
    if not model:
        return ("unknown", 0.0)

    # fastText.predict 返回一个元组，第一个元素是标签列表，第二个是得分列表
    # 标签格式为 ['__label__en']，得分格式为 [0.999]
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    labels, scores = model.predict(cleaned_text)

    # 提取第一个预测结果
    top_label = labels[0]
    top_score = scores[0]

    # 清理标签，移除 '__label__' 前缀
    language_id = top_label.replace('__label__', '')

    # 根据测试要求进行语言代码映射（如果需要）
    # 例如：if language_id == 'zh-TW': language_id = 'zh'
    # 对于 lid.176.bin，通常直接是 'en' 和 'zh'，因此这一步可能不需要
    
    return (language_id, top_score)

# 示例调用
# text_english = "This is a test sentence in English."
# lang, score = run_identify_language(text_english)
# print(f"Text: '{text_english}' -> Language: {lang}, Score: {score}")

# text_chinese = "这是一句测试用的中文句子。"
# lang, score = run_identify_language(text_chinese)
# print(f"Text: '{text_chinese}' -> Language: {lang}, Score: {score}")

def run_mask_emails(text: str) -> tuple[str, int]:
    """
    用 '|||EMAIL_ADDRESS|||' 遮蔽字符串中的电子邮件地址。
    
    Args:
        text: 待处理的输入字符串。

    Returns:
        一个元组，包含遮蔽后的新字符串和被遮蔽的实例数量。
    """
    # 邮箱地址的正则表达式
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # 查找所有匹配项
    matches = re.findall(email_regex, text)
    count = len(matches)
    
    # 替换匹配项
    masked_text = re.sub(email_regex, '|||EMAIL_ADDRESS|||', text)
    
    return masked_text, count

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    用 '|||PHONE_NUMBER|||' 遮蔽字符串中的电话号码。
    """
    # 修正后的正则表达式，更精确地匹配多种美国电话号码格式
    phone_regex = r"""
        (?:1[.\-\s]?)?                    # 可选的 '1' 和分隔符
        (?:\(?(\d{3})\)?[.\-\s]?)         # 匹配区号：(XXX) 或 XXX 或 XXX- 等
        (\d{3})                           # 匹配中间三位数字
        [.\-\s]?                          # 可选的分隔符
        (\d{4})                           # 匹配最后四位数字
    """
    
    # 使用 re.subn 替换并计数，移除 \b 以避免匹配失败
    masked_text, count = re.subn(phone_regex, '|||PHONE_NUMBER|||', text, flags=re.VERBOSE | re.IGNORECASE)
    
    return masked_text, count

def run_mask_ips(text: str) -> tuple[str, int]:
    """
    用 '|||IP_ADDRESS|||' 遮蔽字符串中的 IPv4 地址。
    
    Args:
        text: 待处理的输入字符串。

    Returns:
        一个元组，包含遮蔽后的新字符串和被遮蔽的实例数量。
    """
    # 匹配 IPv4 地址的正则表达式
    ip_regex = r"""
        \b                                     # 单词边界
        (?:                                    # 非捕获组，用于匹配四个数字段
            (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?) # 匹配 0-255 的数字
            \.
        ){3}
        (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
        \b                                     # 单词边界
    """
    
    # 查找所有匹配项
    matches = re.findall(ip_regex, text, re.VERBOSE | re.X)
    count = len(matches)
    
    # 替换匹配项
    masked_text = re.sub(ip_regex, '|||IP_ADDRESS|||', text, flags=re.VERBOSE)
    
    return masked_text, count



# 定义 NSFW 分类器模型路径
NSFW_CLASSIFIER_PATH = "./data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"

# 全局加载 NSFW 模型以提高效率
try:
    nsfw_model = fasttext.load_model(NSFW_CLASSIFIER_PATH)
except ValueError as e:
    print(f"Error loading NSFW fastText model: {e}")
    nsfw_model = None

def run_classify_nsfw(text: str) -> tuple[Any, float]:
    """
    使用预训练的 fastText 模型检测 NSFW 内容。
    
    Args:
        text: 待分类的输入字符串。

    Returns:
        一个元组，包含预测标签（例如 '__label__nsfw'）和置信度分数。
    """
    if nsfw_model is None:
        return "model_not_loaded", 0.0

    # fastText 的 predict 函数一次只处理一行文本，所以需要处理换行符
    cleaned_text = text.replace('\n', ' ').replace('\r', '')

    # 进行预测
    labels, scores = nsfw_model.predict(cleaned_text)

    # 提取标签和分数
    label = labels[0]
    score = scores[0]
    label = label.replace('__label__', '')
    return label, score

# 定义有毒言论分类器模型路径
TOXIC_CLASSIFIER_PATH = "./data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"

# 全局加载有毒言论模型以提高效率
try:
    toxic_model = fasttext.load_model(TOXIC_CLASSIFIER_PATH)
except ValueError as e:
    print(f"Error loading toxic fastText model: {e}")
    toxic_model = None

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    """
    使用预训练的 fastText 模型检测有毒言论。
    
    Args:
        text: 待分类的输入字符串。

    Returns:
        一个元组，包含预测标签（例如 '__label__toxic'）和置信度分数。
    """
    if toxic_model is None:
        return "model_not_loaded", 0.0

    # fastText 的 predict 函数一次只处理一行文本
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    
    # 进行预测
    labels, scores = toxic_model.predict(cleaned_text)
    
    # 提取标签和分数
    label = labels[0]
    score = scores[0]
    label = label.replace('__label__', '')
    return label, score

# 加载训练好的模型
QUALITY_MODEL = None

def load_quality_model(model_path: str = 'quality_classifier.bin'):
    """加载质量分类器模型"""
    global QUALITY_MODEL
    if QUALITY_MODEL is None:
        try:
            QUALITY_MODEL = fasttext.load_model(model_path)
            print(f"Quality model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return QUALITY_MODEL
def run_classify_quality(text: str) -> Tuple[Any, float]:
    """
    对文本进行质量分类，返回标签和置信度分数
    """
    # 加载模型
    model = load_quality_model()
    
    # 预处理文本
    if not text or not isinstance(text, str):
        return 'cc', 0.1
    
    text = text.strip()
    if len(text) < 20:
        return 'cc', 0.3
    
    try:
        # 关键修复：移除所有换行符
        clean_text = text.replace('\n', ' ').replace('\r', ' ')
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text or len(clean_text) < 10:
            return 'cc', 0.2
            
        # 现在可以安全预测
        labels, probabilities = model.predict(clean_text, k=2)
        high_quality_prob = 0.0
        # 预测时，k参数可以设置为1，如果只关心最可能的标签
        labels, probabilities = model.predict(clean_text, k=2) 

        # 核心修复：正确处理labels和probabilities
        high_quality_prob = 0.0
        
        # 遍历所有预测结果，找到high_quality的概率
        for i, label in enumerate(labels):
            if label == '__label__high_quality':
                high_quality_prob = probabilities[i]
                break
        
        print("text is:", clean_text[:500], "score is:", high_quality_prob, "label is:", labels[0])
        
        print("score is:",high_quality_prob,"label is:",label)
        if high_quality_prob >= 0.8:
            return 'wiki', float(high_quality_prob)
        else:
            return 'cc', float(1.0 - high_quality_prob)
            
    except Exception as e:
        print(f"Classification error: {e}")
        return 'cc', 0.1



from nltk.tokenize import word_tokenize



def run_gopher_quality_filter(text: str) -> bool:
    """
    根据Gopher论文的规则过滤低质量文本。

    Args:
        text: 待过滤的输入字符串。

    Returns:
        如果文本通过所有过滤器则返回 True，否则返回 False。
    """
    # 如果输入为空或只有空白，直接返回 False
    if not text.strip():
        return False

    # 1. 单词数量
    words = word_tokenize(text)
    word_count = len(words)
    if not (50 <= word_count <= 100000):
        return False

    # 2. 平均单词长度
    if word_count > 0:
        total_word_length = sum(len(word) for word in words)
        mean_word_length = total_word_length / word_count
        if not (3 <= mean_word_length <= 10):
            return False

    # 3. 超过30%的行以省略号结尾
    lines = text.split('\n')
    ellipsis_line_count = 0
    total_lines = len(lines)
    for line in lines:
        if line.strip().endswith('...'):
            ellipsis_line_count += 1
    
    if total_lines > 0 and (ellipsis_line_count / total_lines) > 0.3:
        return False

    # 4. 至少80%的单词包含一个字母
    alpha_word_count = 0
    for word in words:
        if re.search('[a-zA-Z]', word):
            alpha_word_count += 1
    
    if word_count > 0 and (alpha_word_count / word_count) < 0.8:
        return False

    # 如果所有过滤器都通过，返回 True
    return True



import hashlib

def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    对一组文件执行精确行去重。
    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 第一次遍历：统计每一行的哈希值及其频率
    # 使用哈希来节省内存，键为哈希值，值为频率
    line_counts = {}
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用SHA-256哈希来保证唯一性，哈希值作为键
                line_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    # 第二次遍历：重写每个文件，只保留唯一的行
    for file_path in input_files:
        # 构建输出文件路径，保持原文件名
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_directory, file_name)

        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 再次计算行的哈希值
                line_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                
                # 如果该行在整个语料库中只出现一次，则写入输出文件
                if line_counts.get(line_hash) == 1:
                    outfile.write(line)

    print(f"精确行去重完成，输出文件位于：{output_directory}")







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
    
    Args:
        input_files: 输入文件路径列表
        num_hashes: 计算MinHash签名使用的哈希函数数量
        num_bands: LSH使用的带数量
        ngrams: 计算MinHash签名的n-gram长度（以单词为单位）
        jaccard_threshold: Jaccard相似度阈值
        output_directory: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 存储每个文件的信息
    documents = {}  # file_path -> (normalized_text, ngrams, signature)
    
    # 第1步：读取文件并计算MinHash签名
    print("计算MinHash签名...")
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 标准化文本
            normalized_text = normalize_text(content)
            
            # 提取n-grams
            doc_ngrams = get_ngrams(normalized_text, ngrams)
            
            # 计算MinHash签名
            signature = compute_minhash_signature(doc_ngrams, num_hashes)
            
            documents[file_path] = (normalized_text, doc_ngrams, signature)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    # 第2步：使用LSH识别候选重复项
    print("使用LSH识别候选重复项...")
    band_buckets = defaultdict(set)  # band_hash -> set of file_paths
    
    for file_path, (_, _, signature) in documents.items():
        bands = lsh_bands(signature, num_bands)
        
        for band_idx, band in enumerate(bands):
            # 为每个带创建唯一的桶标识符
            band_hash = hash(f"{band_idx}_{band}")
            band_buckets[band_hash].add(file_path)
    
    # 收集候选重复项对
    candidate_pairs = set()
    for bucket_files in band_buckets.values():
        if len(bucket_files) > 1:
            # 将桶中的所有文件对添加为候选项
            bucket_list = list(bucket_files)
            for i in range(len(bucket_list)):
                for j in range(i + 1, len(bucket_list)):
                    candidate_pairs.add((bucket_list[i], bucket_list[j]))
    
    print(f"发现 {len(candidate_pairs)} 个候选重复项对")
    
    # 第3步：计算候选项的真实Jaccard相似度并识别重复项
    print("计算真实Jaccard相似度...")
    duplicate_pairs = []
    
    for file1, file2 in candidate_pairs:
        _, ngrams1, _ = documents[file1]
        _, ngrams2, _ = documents[file2]
        
        similarity = jaccard_similarity(ngrams1, ngrams2)
        
        if similarity >= jaccard_threshold:
            duplicate_pairs.append((file1, file2, similarity))
    
    print(f"发现 {len(duplicate_pairs)} 个重复项对（阈值 >= {jaccard_threshold}）")
    
    # 第4步：聚类重复文档
    print("聚类重复文档...")
    
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
    for file_path in documents.keys():
        root = find(file_path)
        clusters[root].append(file_path)
    
    # 第5步：从每个聚类中随机选择一个文档保留
    print("选择要保留的文档...")
    files_to_keep = set()
    
    for cluster_files in clusters.values():
        if len(cluster_files) == 1:
            # 单个文件，直接保留
            files_to_keep.add(cluster_files[0])
        else:
            # 多个文件，随机选择一个保留
            selected_file = random.choice(cluster_files)
            files_to_keep.add(selected_file)
            print(f"从聚类 {cluster_files} 中选择保留: {selected_file}")
    
    # 第6步：将保留的文件写入输出目录
    print("写入输出文件...")
    
    for file_path in input_files:
        if file_path in files_to_keep:
            # 构建输出文件路径
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_directory, filename)
            
            try:
                # 复制原始文件内容（不是标准化后的）
                with open(file_path, 'r', encoding='utf-8') as input_file:
                    content = input_file.read()
                
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(content)
                
            except Exception as e:
                print(f"写入文件 {output_path} 时出错: {e}")
    
    print(f"去重完成！保留了 {len(files_to_keep)} 个文件，共 {len(input_files)} 个输入文件")
    print(f"删除了 {len(input_files) - len(files_to_keep)} 个重复文件")


