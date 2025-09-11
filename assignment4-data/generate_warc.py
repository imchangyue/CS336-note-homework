import gzip
import random
import os
import sys

# 文件路径和参数定义
URL_FILE_PATH = "enwiki-20240420-extracted_urls.txt.gz"
OUTPUT_URL_LIST = "subsampled_positive_urls.txt"
POSITIVE_WARC_FILE = "positive_samples.warc"
# 你需要抽样的URLs数量，可以根据你的需求和计算资源进行调整
NUM_SAMPLES = 20000 

def create_subsampled_url_list(url_file_path: str, output_list_path: str, num_samples: int):
    """
    从一个大型URL文件中随机子抽样URL。
    
    Args:
        url_file_path (str): .gz 压缩的 URL 文件路径。
        output_list_path (str): 存储子抽样 URLs 的输出文件路径。
        num_samples (int): 要抽样的 URL 数量。
    """
    print(f"开始从 {url_file_path} 中抽取 {num_samples} 个URLs...")
    
    urls = []
    total_lines = 0
    try:
        # 流式读取以节省内存
        with gzip.open(url_file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                urls.append(line.strip())
                total_lines += 1
                
        if total_lines < num_samples:
            print(f"警告：文件只包含 {total_lines} 个URL，使用全部URLs。")
            subsampled_urls = urls
        else:
            subsampled_urls = random.sample(urls, num_samples)

        with open(output_list_path, 'w') as f:
            f.write('\n'.join(subsampled_urls))
        
        print(f"成功将 {len(subsampled_urls)} 个URLs写入到 {output_list_path}。")
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {url_file_path}。请确保文件已下载到正确的位置。")
        sys.exit(1)

def download_warc_file(url_list_path: str, output_warc_path: str):
    """
    使用 wget 命令下载URLs并保存为WARC格式。
    
    Args:
        url_list_path (str): 包含URLs的文本文件路径。
        output_warc_path (str): 输出的WARC文件路径。
    """
    # 按照你的要求构建 wget 命令
    wget_command = (
        f"wget --timeout=5 -i {url_list_path} "
        f"--warc-file={output_warc_path} -O /dev/null"
    )
    
    print(f"\n开始执行 wget 下载命令，这可能需要一些时间...\n{wget_command}")
    
    # 执行 shell 命令
    try:
        os.system(wget_command)
        print(f"\n下载完成！WARC文件已保存到 {output_warc_path}。")
    except Exception as e:
        print(f"执行 wget 命令时出错: {e}")
        sys.exit(1)

def main():
    # 第1步：创建子抽样URL列表
    create_subsampled_url_list(URL_FILE_PATH, OUTPUT_URL_LIST, NUM_SAMPLES)
    
    # 第2步：下载网页并生成WARC文件
    download_warc_file(OUTPUT_URL_LIST, POSITIVE_WARC_FILE)

if __name__ == "__main__":
    main()