from __future__ import annotations
import os
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re

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

def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


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


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
