import gzip
import random
import os
import re
from bs4 import BeautifulSoup
import fasttext
import warcio.warc
import sys

# 设置全局变量
# 抽样的正例URL数量，可以根据你的计算资源调整
NUM_SAMPLES = 20000 
# Common Crawl数据集的负例，为了简化，我们这里假设从一个模拟文件中获取
# 在实际任务中，你需要从Common Crawl中下载并处理
COMMON_CRAWL_NEGATIVE_FILE = 'common_crawl_negative_samples.txt'
POSITIVE_WARC_FILE = 'positive_samples.warc'
DATASET_FILE = 'quality_dataset.txt'
MODEL_FILE = 'quality_classifier.bin'

def subsample_and_download_urls(url_file_path: str, num_samples: int):
    """
    从维基百科URL文件中随机抽取样本，并下载对应的网页内容。

    Args:
        url_file_path (str): 压缩的URL文件路径。
        num_samples (int): 要抽样的URL数量。
    """
    print("Step 1: Subsampling URLs...")
    urls = []
    try:
        with gzip.open(url_file_path, 'rt') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: URL file not found at {url_file_path}")
        sys.exit(1)

    if len(urls) < num_samples:
        print(f"Warning: The file contains only {len(urls)} URLs, using all of them.")
        subsampled_urls = urls
    else:
        subsampled_urls = random.sample(urls, num_samples)

    with open('subsampled_positive_urls.txt', 'w') as f:
        f.write('\n'.join(subsampled_urls))

    print(f"Saved {len(subsampled_urls)} URLs to subsampled_positive_urls.txt.")
    
    # 使用wget下载网页，并以WARC格式保存
    # 注意：这个命令需要在shell中执行，并且需要安装wget
    wget_command = (
        f"wget --timeout=5 -i subsampled_positive_urls.txt "
        f"--warc-file={POSITIVE_WARC_FILE} -O /dev/null"
    )
    print(f"Downloading URLs using wget command:\n{wget_command}")
    # os.system(wget_command)
    print("Download finished. WARC file created.")

def extract_and_clean_text_from_warc(warc_file_path: str) -> list[str]:
    """
    从WARC文件中提取并清洗文本内容。
    
    Args:
        warc_file_path (str): WARC文件路径。

    Returns:
        list[str]: 清洗后的文本列表。
    """
    print("Step 2: Extracting and cleaning text from WARC...")
    texts = []
    try:
        with open(warc_file_path, 'rb') as stream:
            reader = warcio.warc.WARCReader(stream)
            for record in reader:
                if record.rec_type == 'response' and record.http_headers:
                    # 仅处理HTML内容
                    content_type = record.http_headers.get_header('Content-Type')
                    if content_type and 'text/html' in content_type:
                        html_content = record.content_stream().read()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        # 移除脚本和样式
                        for script_or_style in soup(['script', 'style']):
                            script_or_style.decompose()
                        
                        text = soup.get_text(separator=' ', strip=True)
                        # 移除多余的空格和换行
                        text = re.sub(r'\s+', ' ', text)
                        texts.append(text)
    except FileNotFoundError:
        print(f"Error: WARC file not found at {warc_file_path}. Please check your download step.")
        sys.exit(1)
    
    print(f"Extracted {len(texts)} text samples.")
    return texts

def prepare_fasttext_dataset(positive_texts: list[str], negative_texts: list[str], output_file_path: str):
    """
    将正负例文本格式化为fastText的训练集格式。

    Args:
        positive_texts (list[str]): 正例文本列表。
        negative_texts (list[str]): 负例文本列表。
        output_file_path (str): 输出文件路径。
    """
    print("Step 3: Preparing dataset for fastText...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for text in positive_texts:
            f.write(f"__label__high_quality {text}\n")
        
        # 为了平衡数据集，只使用与正例数量相当的负例
        num_negatives = min(len(positive_texts), len(negative_texts))
        for text in negative_texts[:num_negatives]:
            f.write(f"__label__low_quality {text}\n")
    
    print(f"Dataset saved to {output_file_path}.")

def train_classifier_model(dataset_file_path: str, model_output_path: str):
    """
    使用fastText训练分类器模型。

    Args:
        dataset_file_path (str): 训练集文件路径。
        model_output_path (str): 模型输出路径。
    """
    print("Step 4: Training the fastText model...")
    # 可以根据你的需求调整参数
    # wordNgrams: 2-gram 通常效果不错
    # epoch: 迭代次数，适当增加可以提升性能
    model = fasttext.train_supervised(
        input=dataset_file_path,
        epoch=25,
        wordNgrams=2,
        dim=100,
        loss='softmax'
    )
    model.save_model(model_output_path)
    print(f"Model saved to {model_output_path}")

def main():
    # 维基百科URL文件路径
    wiki_urls_path = 'enwiki-20240420-extracted_urls.txt.gz'
    
    # 模拟获取负例数据，在实际中你需要从Common Crawl中处理
    # ⚠️ 提示：为了运行本脚本，请先手动创建一个 common_crawl_negative_samples.txt 文件
    # 里面包含一些低质量的随机文本，每行一个样本。
    if not os.path.exists(COMMON_CRAWL_NEGATIVE_FILE):
        print("Creating a dummy negative samples file for demonstration...")
        with open(COMMON_CRAWL_NEGATIVE_FILE, 'w') as f:
            f.write("A bunch of random words and broken sentences. This is a low-quality text example. \n")
            f.write("Another low quality example, with no real meaning. It's just junk content.\n")
        print(f"Please fill {COMMON_CRAWL_NEGATIVE_FILE} with more data for better results.")
    
    with open(COMMON_CRAWL_NEGATIVE_FILE, 'r') as f:
        negative_texts = [line.strip() for line in f]

    # 1. 抽样并下载
    subsample_and_download_urls(wiki_urls_path, NUM_SAMPLES)

    # 2. 从WARC文件中提取文本
    positive_texts = extract_and_clean_text_from_warc(POSITIVE_WARC_FILE)
    
    # 3. 准备数据集
    prepare_fasttext_dataset(positive_texts, negative_texts, DATASET_FILE)
    
    # 4. 训练模型
    train_classifier_model(DATASET_FILE, MODEL_FILE)
    
    print("\nTraining complete! The model is ready for use in part (b).")

if __name__ == "__main__":
    main()