import os
import re
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext

# 在全局作用域加载模型以提高效率，避免重复加载
try:
    model = fasttext.load_model('/data/classifiers/lid.176.bin')
    print("fastText model loaded successfully.")
except ValueError as e:
    print(f"Error loading fastText model: {e}")
    model = None

# 使用你的 run_extract_text_from_html_bytes 函数
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        # 修复：resiliparse 的 detect_encoding() 函数在某些版本中可能只返回一个字符串
        encoding = detect_encoding(html_bytes)
        html_string = html_bytes.decode(encoding, errors='replace')
        
        plain_text = extract_plain_text(html_string)
        return plain_text
    except Exception as e:
        print(f"处理HTML时出错: {e}")
        return None

# 新增的语言识别函数，用于处理包含换行符的文本
def run_identify_language(text: str) -> tuple[str, float]:
    """
    使用 fastText 模型识别文本语言。
    """
    if model is None:
        raise ValueError("fastText model is not loaded.")

    # 修复：移除所有换行符以避免 fastText 抛出 ValueError
    cleaned_text = text.replace('\n', ' ').replace('\r', '')

    # 调用 fastText 模型进行预测
    labels, scores = model.predict(cleaned_text)

    # 提取标签和分数
    predicted_language = labels[0].replace('__label__', '') # 移除 fasttext 前缀
    score = scores[0]

    return predicted_language, score

# PII 遮蔽函数
def run_mask_emails(text: str) -> tuple[str, int]:
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    masked_text, count = re.subn(email_regex, '|||EMAIL_ADDRESS|||', text)
    return masked_text, count

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_regex = r"""
        (?:1[.\-\s]?)?                    
        (?:\(?(\d{3})\)?[.\-\s]?)         
        (\d{3})                           
        [.\-\s]?                          
        (\d{4})                           
    """
    masked_text, count = re.subn(phone_regex, '|||PHONE_NUMBER|||', text, flags=re.VERBOSE | re.IGNORECASE)
    return masked_text, count

def run_mask_ips(text: str) -> tuple[str, int]:
    ip_regex = r"""
        \b                                     
        (?:                                    
            (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?) 
            \.
        ){3}
        (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
        \b                                     
    """
    masked_text, count = re.subn(ip_regex, '|||IP_ADDRESS|||', text, flags=re.VERBOSE | re.X)
    return masked_text, count

# 主函数
def main():
    warc_file_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

    if not os.path.exists(warc_file_path):
        print(f"错误: 找不到文件 {warc_file_path}")
        return

    record_count = 0
    with open(warc_file_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type != WarcRecordType.response:
                continue
            content_type = record.http_headers.get('Content-Type', '').split(';')[0].lower()
            if 'text/html' not in content_type:
                continue
            
            html_bytes = record.reader.read()
            text = run_extract_text_from_html_bytes(html_bytes)
            
            if text:
                try:
                    # 对文本进行 PII 遮蔽
                    masked_text, email_count = run_mask_emails(text)
                    masked_text, phone_count = run_mask_phone_numbers(masked_text)
                    masked_text, ip_count = run_mask_ips(masked_text)

                    if email_count > 0 or phone_count > 0 or ip_count > 0:
                        print(f"URL: {record.headers['WARC-Target-URI']}")
                        print(f"被遮蔽的实体数量：邮箱={email_count}, 电话={phone_count}, IP={ip_count}")
                        print("原始文本片段：")
                        print(text[:200].strip() + "...")
                        print("遮蔽后文本片段：")
                        print(masked_text[:200].strip() + "...\n")
                        record_count += 1
                        if record_count >= 10:
                            break
                except Exception as e:
                    print(f"处理记录时出错: {e}")
            
    print(f"\n总共处理了 {record_count} 个进行了PII遮蔽的HTML记录。")

if __name__ == "__main__":
    main()