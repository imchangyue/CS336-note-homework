# generate_negative_samples.py

import os
import re
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from nltk.tokenize import word_tokenize
import fasttext

# 定义模型路径，确保这些路径在你的环境中是正确的
# ⚠️ 注意: 请根据你的实际路径调整
NSFW_CLASSIFIER_PATH = "./data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_CLASSIFIER_PATH = "./data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"
LANG_CLASSIFIER_PATH = "/data/classifiers/lid.176.bin"

# 定义输出文件路径
COMMON_CRAWL_NEGATIVE_FILE = "common_crawl_negative_samples.txt"

# ----------------- 全局加载模型 -----------------
try:
    lang_model = fasttext.load_model(LANG_CLASSIFIER_PATH)
    print("fastText language model loaded successfully.")
except ValueError as e:
    print(f"Error loading fastText language model: {e}")
    lang_model = None

try:
    toxic_model = fasttext.load_model(TOXIC_CLASSIFIER_PATH)
    print("fastText toxic speech model loaded successfully.")
except ValueError as e:
    print(f"Error loading toxic fastText model: {e}")
    toxic_model = None

try:
    nsfw_model = fasttext.load_model(NSFW_CLASSIFIER_PATH)
    print("fastText NSFW model loaded successfully.")
except ValueError as e:
    print(f"Error loading NSFW fastText model: {e}")
    nsfw_model = None

# ----------------- 文本处理函数 (来自您的代码) -----------------
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        encoding = detect_encoding(html_bytes)
        html_string = html_bytes.decode(encoding, errors='replace')
        plain_text = extract_plain_text(html_string)
        return plain_text
    except Exception as e:
        return None

def run_identify_language(text: str) -> tuple[str, float]:
    if lang_model is None:
        raise ValueError("fastText language model is not loaded.")
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    labels, scores = lang_model.predict(cleaned_text)
    predicted_language = labels[0].replace('__label__', '')
    score = scores[0]
    return predicted_language, score

def run_classify_nsfw(text: str) -> tuple[Any, float]:
    if nsfw_model is None:
        return "model_not_loaded", 0.0
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    labels, scores = nsfw_model.predict(cleaned_text)
    label = labels[0]
    score = scores[0]
    label = label.replace('__label__', '')
    return label, score

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    if toxic_model is None:
        return "model_not_loaded", 0.0
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    labels, scores = toxic_model.predict(cleaned_text)
    label = labels[0]
    score = scores[0]
    label = label.replace('__label__', '')
    return label, score
    
def run_gopher_quality_filter(text: str) -> bool:
    if not text.strip():
        return False
    words = word_tokenize(text)
    word_count = len(words)
    if not (50 <= word_count <= 100000):
        return False
    if word_count > 0:
        total_word_length = sum(len(word) for word in words)
        mean_word_length = total_word_length / word_count
        if not (3 <= mean_word_length <= 10):
            return False
    lines = text.split('\n')
    ellipsis_line_count = sum(1 for line in lines if line.strip().endswith('...'))
    total_lines = len(lines)
    if total_lines > 0 and (ellipsis_line_count / total_lines) > 0.3:
        return False
    alpha_word_count = sum(1 for word in words if re.search('[a-zA-Z]', word))
    if word_count > 0 and (alpha_word_count / word_count) < 0.8:
        return False
    return True

# ----------------- 主逻辑 -----------------
def main():
    # ⚠️ 提示：请将此路径替换为你实际下载的 Common Crawl warc 文件路径
    warc_file_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

    if not os.path.exists(warc_file_path):
        print(f"错误: 找不到文件 {warc_file_path}")
        return

    negative_samples_count = 0
    total_processed = 0
    # 设置一个上限，以避免生成过大的文件
    MAX_NEGATIVE_SAMPLES = 50000 

    print(f"开始从 {warc_file_path} 中提取负例样本...")
    
    with open(warc_file_path, "rb") as stream, open(COMMON_CRAWL_NEGATIVE_FILE, "w", encoding="utf-8") as out_file:
        for record in ArchiveIterator(stream):
            if record.record_type != WarcRecordType.response:
                continue
            content_type = record.http_headers.get('Content-Type', '').split(';')[0].lower()
            if 'text/html' not in content_type:
                continue
            
            html_bytes = record.reader.read()
            text = run_extract_text_from_html_bytes(html_bytes)
            
            if not text:
                continue

            # 过滤短文本
            if len(text.split()) < 50:
                continue
            
            total_processed += 1
            
            # === 判断是否为低质量负例的核心逻辑 ===
            is_low_quality = False
            
            # 规则1: 不通过Gopher质量过滤器
            if not run_gopher_quality_filter(text):
                is_low_quality = True
            
            # 规则2: NSFW或有毒言论
            nsfw_label, nsfw_score = run_classify_nsfw(text)
            toxic_label, toxic_score = run_classify_toxic_speech(text)
            
            if (nsfw_label == 'nsfw' and nsfw_score > 0.5) or \
               (toxic_label == 'toxic' and toxic_score > 0.5):
                is_low_quality = True
                
            # 规则3: 非英语文本
            lang, _ = run_identify_language(text)
            if lang != 'en':
                is_low_quality = True
            
            # 如果被标记为低质量，则写入文件
            if is_low_quality:
                out_file.write(text.strip().replace('\n', ' ') + "\n")
                negative_samples_count += 1
                
            if negative_samples_count >= MAX_NEGATIVE_SAMPLES:
                print(f"已达到 {MAX_NEGATIVE_SAMPLES} 个负例样本上限，停止处理。")
                break
    
    print("-" * 50)
    print(f"总共处理了 {total_processed} 个文档。")
    print(f"已成功将 {negative_samples_count} 个负例样本写入到 {COMMON_CRAWL_NEGATIVE_FILE}。")
    print("现在你可以使用这个文件作为训练质量分类器的负例数据。")

if __name__ == "__main__":
    main()