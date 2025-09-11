import nltk
import os

# 定义你希望下载NLTK数据的目录
# 确保这个目录存在，如果不存在，请先创建它
nltk_data_path = '/home/code_backup/code/cs336/assignment4-data/.venv/nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# 将NLTK数据目录添加到搜索路径中
# 这会告诉NLTK从哪里寻找和下载数据
nltk.data.path.append(nltk_data_path)

# 现在，执行下载
# NLTK会把'punkt_tab'下载到你指定的路径
nltk.download('punkt_tab', download_dir=nltk_data_path)

# 验证是否成功下载
try:
    nltk.data.find('tokenizers/punkt_tab/english')
    print("punkt_tab 下载成功并已找到！")
except nltk.downloader.DownloadError:
    print("下载失败或未找到。")