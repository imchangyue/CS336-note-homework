import gzip
import random
import os
import re
from bs4 import BeautifulSoup
import fasttext
from warcio import ArchiveIterator  # 修改导入方式
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Some characters could not be decoded')

# 设置全局变量
NUM_SAMPLES = 20000
COMMON_CRAWL_NEGATIVE_FILE = 'common_crawl_negative_samples.txt'
POSITIVE_WARC_FILE = 'positive_samples.warc.warc'  # 修正文件名
DATASET_FILE = 'quality_dataset.txt'
MODEL_FILE = 'quality_classifier.bin'

def extract_and_clean_text_from_warc(warc_file_path: str) -> list[str]:
    """
    从WARC文件中提取并清洗文本内容。
    """
    print("Step 1: Extracting and cleaning text from WARC...")
    texts = []
    try:
        with open(warc_file_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    # 检查内容类型
                    content_type = record.http_headers.get_header('Content-Type', '')
                    if 'text/html' in content_type:
                        try:
                            html_content = record.content_stream().read()
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            # 移除脚本和样式
                            for script_or_style in soup(['script', 'style', 'nav', 'footer', 'header']):
                                script_or_style.decompose()
                            
                            # 获取主要文本内容
                            text = soup.get_text(separator=' ', strip=True)
                            # 清理文本
                            text = re.sub(r'\s+', ' ', text)
                            text = text.strip()
                            #print("text:",text)
                            
                            if text and len(text) > 100:  # 过滤太短的文本
                                texts.append(text)
                                
                        except Exception as e:
                            print(f"Error processing record: {e}")
                            continue
                            
    except FileNotFoundError:
        print(f"Error: WARC file not found at {warc_file_path}")
        sys.exit(1)
    
    print(f"Extracted {len(texts)} text samples.")
    return texts

def load_negative_samples(negative_file_path: str) -> list[str]:
    """
    加载负例样本
    """
    print("Step 2: Loading negative samples...")
    negative_texts = []
    try:
        with open(negative_file_path, 'r', encoding='utf-8') as f:
            negative_texts = [line.strip() for line in f if line.strip()]
            # print("negative sample:",negative_texts[:5])
    except FileNotFoundError:
        print(f"Warning: Negative samples file not found at {negative_file_path}")
        # 创建一些模拟负例
        negative_texts = [
            "Random gibberish text with no meaning or coherence.",
            "Low quality content filled with spam and advertisements.",
            "Broken sentences and incomplete thoughts without structure.",
            "Repetitive nonsense text that provides no valuable information.",
            "Poorly written content with many grammatical errors and typos."
        ] * 1000  # 复制一些样本
    
    print(f"Loaded {len(negative_texts)} negative samples.")
    return negative_texts

def prepare_fasttext_dataset(positive_texts: list[str], negative_texts: list[str], output_file_path: str):
    """
    准备fastText训练数据集，处理编码问题
    """
    print("Step 3: Preparing dataset for fastText...")
    
    # 确保正负例数量平衡
    num_samples = min(len(positive_texts), len(negative_texts))
    positive_texts = positive_texts[:num_samples]
    negative_texts = negative_texts[:num_samples]
    
    def clean_text(text):
        """清理文本中的非法UTF-8字符"""
        if not text:
            return ""
        # 方法1：移除代理对字符
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        # 方法2：进一步清理（可选）
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # 移除控制字符
        text = re.sub(r'\s+', ' ', text).strip()  # 规范化空格
        return text
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 处理正例
        for i, text in enumerate(positive_texts):
            cleaned_text = clean_text(text)
            if cleaned_text and len(cleaned_text) > 10:  # 确保文本有效
                f.write(f"__label__high_quality {cleaned_text}\n")
            else:
                print(f"Skipping positive sample {i} due to cleaning issues")
        
        # 处理负例
        for i, text in enumerate(negative_texts):
            cleaned_text = clean_text(text)
            if cleaned_text and len(cleaned_text) > 10:  # 确保文本有效
                f.write(f"__label__low_quality {cleaned_text}\n")
            else:
                print(f"Skipping negative sample {i} due to cleaning issues")
    
    print(f"Dataset saved to {output_file_path}")
    print(f"Positive samples: {len(positive_texts)}, Negative samples: {len(negative_texts)}")

    
def train_classifier_model(dataset_file_path: str, model_output_path: str):
    """
    训练fastText分类器
    """
    print("Step 4: Training the fastText model...")
    
    try:
        model = fasttext.train_supervised(
            input=dataset_file_path,
            epoch=10,  # 减少epoch数以加快训练
            wordNgrams=2,
            dim=50,    # 减少维度以节省内存
            loss='softmax'
        )
        
        # 测试模型性能
        result = model.test(dataset_file_path)
        print(f"Model accuracy: {result[1]:.4f}, Precision: {result[2]:.4f}")
        
        model.save_model(model_output_path)
        print(f"Model saved to {model_output_path}")
        
    except Exception as e:
        print(f"Error training model: {e}")

def main():
    print("Starting quality classifier training...")
    
    # 2. 加载负例文本
    negative_texts = load_negative_samples(COMMON_CRAWL_NEGATIVE_FILE)
    # 1. 从WARC文件中提取正例文本
    positive_texts = extract_and_clean_text_from_warc(POSITIVE_WARC_FILE)
    
    # 3. 准备训练数据集
    prepare_fasttext_dataset(positive_texts, negative_texts, DATASET_FILE)
    
    # 4. 训练模型
    train_classifier_model(DATASET_FILE, MODEL_FILE)
    
    print("\nTraining complete! The model is ready for use.")

if __name__ == "__main__":
    main()