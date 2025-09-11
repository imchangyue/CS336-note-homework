import os
import re
import random
from typing import Any
import nltk
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from nltk.tokenize import word_tokenize

# 确保 NLTK 'punkt' 分词器已下载，这是 Gopher 过滤器所必需的。
# 如果你之前没有下载，请运行下面的代码行。
# nltk.download('punkt')

# 文本提取函数
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    从 HTML 字节中提取纯文本。
    """
    try:
        encoding = detect_encoding(html_bytes)
        html_string = html_bytes.decode(encoding, errors='replace')
        plain_text = extract_plain_text(html_string)
        return plain_text
    except Exception as e:
        # 即使处理失败也不打印错误，只返回 None
        return None

# Gopher 质量过滤器函数
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

    try:
        # 1. 单词数量
        words = word_tokenize(text)
        word_count = len(words)
        if not (50 <= word_count <= 100000):
            return False
    except LookupError:
        # 如果 nltk.download('punkt') 没有运行，会发生此错误。
        print("错误：NLTK 'punkt' 分词器未下载。请先运行 'nltk.download(\"punkt\")'")
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

# 主函数
def main():
    warc_file_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

    if not os.path.exists(warc_file_path):
        print(f"错误: 找不到文件 {warc_file_path}")
        return

    all_records = []
    print("正在从 WARC 文件中加载记录，请稍候...")
    with open(warc_file_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                content_type = record.http_headers.get('Content-Type', '').split(';')[0].lower()
                if 'text/html' in content_type:
                    # 调用 freeze() 来保留记录的内容，防止ReaderStaleError
                    record.freeze()
                    all_records.append(record)

    if len(all_records) < 20:
        print("警告: 找到的 HTML 文档少于 20 个。将处理所有找到的文档。")
        samples = all_records
    else:
        samples = random.sample(all_records, 20)

    print("-" * 50)
    print("开始对 20 个随机文档运行Gopher质量过滤器...")
    for i, record in enumerate(samples):
        # 现在 record.reader 可以正常工作，因为在添加到列表前已经调用了 freeze()
        html_bytes = record.reader.read()
        text = run_extract_text_from_html_bytes(html_bytes)
        
        if text:
            is_high_quality = run_gopher_quality_filter(text)
            
            print(f"\n--- 样本 {i+1}/20 ---")
            print(f"URL: {record.headers['WARC-Target-URI']}")
            print(f"Gopher 过滤器预测: {'高质量' if is_high_quality else '低质量'}")
            print("--- 文本片段（供人工判断）---")
            print(text[:500].strip() + "...")
    
    print("-" * 50)
    print("分析完成。")

if __name__ == "__main__":
    main()
