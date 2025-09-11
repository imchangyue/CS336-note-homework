import os
import re
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext

# 在全局作用域加载所有模型以提高效率，避免重复加载
try:
    # 语言识别模型
    lang_model = fasttext.load_model('/data/classifiers/lid.176.bin')
    print("fastText language model loaded successfully.")
except ValueError as e:
    print(f"Error loading fastText language model: {e}")
    lang_model = None

try:
    # 有毒言论分类器模型
    toxic_model = fasttext.load_model("./data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin")
    print("fastText toxic speech model loaded successfully.")
except ValueError as e:
    print(f"Error loading toxic fastText model: {e}")
    toxic_model = None

try:
    # NSFW 分类器模型
    nsfw_model = fasttext.load_model("./data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin")
    print("fastText NSFW model loaded successfully.")
except ValueError as e:
    print(f"Error loading NSFW fastText model: {e}")
    nsfw_model = None


# 文本提取函数
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        encoding = detect_encoding(html_bytes)
        html_string = html_bytes.decode(encoding, errors='replace')
        plain_text = extract_plain_text(html_string)
        return plain_text
    except Exception as e:
        # print(f"处理HTML时出错: {e}")
        return None

# 语言识别函数
def run_identify_language(text: str) -> tuple[str, float]:
    if lang_model is None:
        raise ValueError("fastText language model is not loaded.")
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    labels, scores = lang_model.predict(cleaned_text)
    predicted_language = labels[0].replace('__label__', '')
    score = scores[0]
    return predicted_language, score

# NSFW 分类函数
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

# 有毒言论分类函数
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

# 主函数
def main():
    warc_file_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

    if not os.path.exists(warc_file_path):
        print(f"错误: 找不到文件 {warc_file_path}")
        return

    harmful_count = 0
    total_processed = 0

    with open(warc_file_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type != WarcRecordType.response:
                continue
            content_type = record.http_headers.get('Content-Type', '').split(';')[0].lower()
            if 'text/html' not in content_type:
                continue
            
            html_bytes = record.reader.read()
            text = run_extract_text_from_html_bytes(html_bytes)
            
            if text and len(text.split()) > 10:
                total_processed += 1
                nsfw_label, nsfw_score = run_classify_nsfw(text)
                toxic_label, toxic_score = run_classify_toxic_speech(text)

                # 修正后的条件判断
                if nsfw_label == 'nsfw' and nsfw_score > 0.5 or toxic_label == 'toxic' and toxic_score > 0.5:
                    harmful_count += 1
                    print("-" * 50)
                    print(f"URL: {record.headers['WARC-Target-URI']}")
                    print(f"NSFW 预测: {nsfw_label} (置信度: {nsfw_score:.4f})")
                    print(f"有毒言论预测: {toxic_label} (置信度: {toxic_score:.4f})")
                    print("文本片段:")
                    print(text[:200].strip() + "...\n")
                
                if harmful_count >= 10:
                    break
    
    print("-" * 50)
    print(f"总共分析了 {total_processed} 个文档。")
    print(f"其中 {harmful_count} 个被标记为有害内容。")
    print(f"有害文档比例约为: {harmful_count / total_processed * 100:.2f} %")


if __name__ == "__main__":
    main()