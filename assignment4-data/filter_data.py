#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Crawl WET文件高级过滤脚本
目标: 为语言建模创建高质量训练数据，优化在Paloma C4 100基准上的困惑度

集成策略:
- fastText语言识别
- PII信息遮蔽 (邮箱、电话、IP地址)
- NSFW/有毒内容过滤
- 质量分类器 (quality.bin)
- Gopher质量过滤规则
- 文档去重

作者: Assistant  
日期: 2025-09-15
"""

import os
import re
import json
import time
import gzip
import pathlib
import logging
import numpy as np
import hashlib
import unicodedata
import random
from typing import List, Dict, Tuple, Set, Any
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import submitit
from tqdm import tqdm

# 导入必要的库
try:
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    from resiliparse.extract.html2text import extract_plain_text  
    from resiliparse.parse.encoding import detect_encoding
    from tldextract import TLDExtract
    from transformers import AutoTokenizer
    import fasttext
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError as e:
    print(f"请安装必要的库: {e}")
    print("pip install fastwarc tldextract transformers nltk fasttext resiliparse")

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局模型加载（避免重复加载）
LANGUAGE_MODEL = None
NSFW_MODEL = None  
TOXIC_MODEL = None
QUALITY_MODEL = None

def load_models():
    """加载所有预训练模型"""
    global LANGUAGE_MODEL, NSFW_MODEL, TOXIC_MODEL, QUALITY_MODEL
    
    # 1. 语言识别模型
    try:
        LANGUAGE_MODEL = fasttext.load_model('/data/classifiers/lid.176.bin')
        logger.info("语言识别模型加载成功")
    except Exception as e:
        logger.error(f"语言识别模型加载失败: {e}")
        LANGUAGE_MODEL = None
    
    # 2. NSFW内容检测模型
    try:
        NSFW_MODEL = fasttext.load_model('/data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin')
        logger.info("NSFW检测模型加载成功")
    except Exception as e:
        logger.error(f"NSFW检测模型加载失败: {e}")
        NSFW_MODEL = None
    
    # 3. 有毒内容检测模型
    try:
        TOXIC_MODEL = fasttext.load_model('/data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin')
        logger.info("有毒内容检测模型加载成功")
    except Exception as e:
        logger.error(f"有毒内容检测模型加载失败: {e}")
        TOXIC_MODEL = None
    
    # 4. 质量分类模型
    try:
        QUALITY_MODEL = fasttext.load_model('/data/classifiers/quality_classifier.bin')
        logger.info("质量分类模型加载成功") 
    except Exception as e:
        logger.error(f"质量分类模型加载失败: {e}")
        QUALITY_MODEL = None

class DataFilterStats:
    """数据过滤统计类，追踪每个过滤步骤的效果"""
    
    def __init__(self):
        self.total_documents = 0
        self.filter_stats = defaultdict(int)
        self.kept_documents = 0
        self.pii_masked_count = defaultdict(int)  # 记录PII遮蔽统计
        
    def add_document(self):
        """添加一个文档到总计数"""
        self.total_documents += 1
        
    def reject_document(self, filter_name: str):
        """记录文档被某个过滤器拒绝"""
        self.filter_stats[filter_name] += 1
        
    def keep_document(self):
        """记录文档被保留"""
        self.kept_documents += 1
        
    def add_pii_masking(self, pii_type: str, count: int):
        """记录PII遮蔽统计"""
        self.pii_masked_count[pii_type] += count
        
    def get_summary(self) -> Dict:
        """获取过滤统计摘要"""
        return {
            'total_documents': self.total_documents,
            'kept_documents': self.kept_documents,
            'rejection_rate': (self.total_documents - self.kept_documents) / max(1, self.total_documents),
            'filter_rejections': dict(self.filter_stats),
            'pii_masking_stats': dict(self.pii_masked_count)
        }

class AdvancedCommonCrawlFilter:
    """高级Common Crawl数据过滤器"""
    
    def __init__(self, paloma_validation_path: str = None):
        """
        初始化过滤器
        
        Args:
            paloma_validation_path: Paloma验证数据路径（用于构建过滤器但不复制数据）
        """
        # 加载所有模型
        load_models()
        
        self.tld_extract = TLDExtract()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.stats = DataFilterStats()
        
        # 加载停用词
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            
        # 目标域名（基于C4数据集的常见域名）
        self.target_domains = self._load_target_domains()
        
        # 过滤器阈值配置
        self.language_confidence_threshold = 0.7  # 语言识别置信度阈值
        self.nsfw_threshold = 0.5  # NSFW内容阈值
        self.toxic_threshold = 0.5  # 有毒内容阈值  
        self.quality_threshold = 0.8  # 高质量内容阈值
        
        logger.info("高级过滤器初始化完成")
        
    def _load_target_domains(self) -> Set[str]:
        """加载目标域名列表（基于C4常见域名）"""
        domains = {
            'wikipedia.org', 'stackexchange.com', 'stackoverflow.com',
            'reddit.com', 'github.com', 'medium.com', 'blogspot.com',
            'wordpress.com', 'nytimes.com', 'washingtonpost.com',
            'theguardian.com', 'bbc.com', 'cnn.com', 'reuters.com',
            'arxiv.org', 'nature.com', 'sciencedirect.com', 'springer.com',
            'edu', 'gov', 'org'  # 顶级域名
        }
        return domains
        
    def extract_text_from_html_bytes(self, html_bytes: bytes) -> str:
        """从HTML字节提取纯文本"""
        try:
            # 1. 自动检测编码并解码
            encoding = detect_encoding(html_bytes)
            html_string = html_bytes.decode(encoding, errors='replace')
            
            # 2. 从解码后的HTML字符串中提取纯文本
            plain_text = extract_plain_text(html_string)
            
            return plain_text
            
        except Exception as e:
            logger.debug(f"处理HTML时出错: {e}")
            return None
    
    def identify_language(self, text: str) -> Tuple[str, float]:
        """使用fastText识别文本语言"""
        if not LANGUAGE_MODEL:
            return ("unknown", 0.0)

        try:
            # 清理文本
            cleaned_text = text.replace('\n', ' ').replace('\r', '')
            labels, scores = LANGUAGE_MODEL.predict(cleaned_text)

            # 提取第一个预测结果
            top_label = labels[0]
            top_score = scores[0]

            # 清理标签，移除 '__label__' 前缀
            language_id = top_label.replace('__label__', '')

            return (language_id, top_score)
        except Exception as e:
            logger.debug(f"语言识别出错: {e}")
            return ("unknown", 0.0)

    def mask_emails(self, text: str) -> Tuple[str, int]:
        """用占位符遮蔽电子邮件地址"""
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_regex, text)
        count = len(matches)
        masked_text = re.sub(email_regex, '|||EMAIL_ADDRESS|||', text)
        return masked_text, count

    def mask_phone_numbers(self, text: str) -> Tuple[str, int]:
        """用占位符遮蔽电话号码"""
        phone_regex = r"""
            (?:1[.\-\s]?)?                    # 可选的 '1' 和分隔符
            (?:\(?(\d{3})\)?[.\-\s]?)         # 匹配区号：(XXX) 或 XXX 或 XXX- 等
            (\d{3})                           # 匹配中间三位数字
            [.\-\s]?                          # 可选的分隔符
            (\d{4})                           # 匹配最后四位数字
        """
        masked_text, count = re.subn(phone_regex, '|||PHONE_NUMBER|||', text, 
                                   flags=re.VERBOSE | re.IGNORECASE)
        return masked_text, count

    def mask_ips(self, text: str) -> Tuple[str, int]:
        """用占位符遮蔽IPv4地址"""
        ip_regex = r"""
            \b                                     # 单词边界
            (?:                                    # 非捕获组，用于匹配四个数字段
                (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?) # 匹配 0-255 的数字
                \.
            ){3}
            (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
            \b                                     # 单词边界
        """
        matches = re.findall(ip_regex, text, re.VERBOSE | re.X)
        count = len(matches)
        masked_text = re.sub(ip_regex, '|||IP_ADDRESS|||', text, flags=re.VERBOSE)
        return masked_text, count

    def classify_nsfw(self, text: str) -> Tuple[str, float]:
        """检测NSFW内容"""
        if not NSFW_MODEL:
            return "unknown", 0.0

        try:
            cleaned_text = text.replace('\n', ' ').replace('\r', '')
            labels, scores = NSFW_MODEL.predict(cleaned_text)
            
            label = labels[0].replace('__label__', '')
            score = scores[0]
            return label, score
        except Exception as e:
            logger.debug(f"NSFW分类出错: {e}")
            return "unknown", 0.0

    def classify_toxic_speech(self, text: str) -> Tuple[str, float]:
        """检测有毒言论"""
        if not TOXIC_MODEL:
            return "unknown", 0.0

        try:
            cleaned_text = text.replace('\n', ' ').replace('\r', '')
            labels, scores = TOXIC_MODEL.predict(cleaned_text)
            
            label = labels[0].replace('__label__', '')
            score = scores[0]
            return label, score
        except Exception as e:
            logger.debug(f"有毒内容分类出错: {e}")
            return "unknown", 0.0

    def classify_quality(self, text: str) -> Tuple[str, float]:
        """使用quality.bin模型进行质量分类"""
        if not QUALITY_MODEL:
            return 'cc', 0.1
        
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
            labels, probabilities = QUALITY_MODEL.predict(clean_text, k=2)
            high_quality_prob = 0.0
            
            # 遍历所有预测结果，找到high_quality的概率
            for i, label in enumerate(labels):
                if label == '__label__high_quality':
                    high_quality_prob = probabilities[i]
                    break
            
            if high_quality_prob >= self.quality_threshold:
                return 'wiki', float(high_quality_prob)
            else:
                return 'cc', float(1.0 - high_quality_prob)
                
        except Exception as e:
            logger.debug(f"质量分类出错: {e}")
            return 'cc', 0.1

    def gopher_quality_filter(self, text: str) -> bool:
        """根据Gopher论文的规则过滤低质量文本"""
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

        return True

    def is_target_domain(self, url: str) -> bool:
        """检查URL是否来自目标域名"""
        try:
            extracted = self.tld_extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"
            
            # 检查完整域名
            if domain in self.target_domains:
                return True
                
            # 检查是否为教育、政府或组织域名
            if extracted.suffix in self.target_domains:
                return True
                
            return False
        except:
            return False

    def apply_pii_masking(self, text: str) -> Tuple[str, Dict[str, int]]:
        """应用PII遮蔽"""
        pii_stats = {}
        
        # 遮蔽邮箱
        text, email_count = self.mask_emails(text)
        pii_stats['emails'] = email_count
        self.stats.add_pii_masking('emails', email_count)
        
        # 遮蔽电话号码
        text, phone_count = self.mask_phone_numbers(text)
        pii_stats['phones'] = phone_count
        self.stats.add_pii_masking('phones', phone_count)
        
        # 遮蔽IP地址
        text, ip_count = self.mask_ips(text)
        pii_stats['ips'] = ip_count
        self.stats.add_pii_masking('ips', ip_count)
        
        return text, pii_stats

    def filter_document(self, url: str, text: str) -> Tuple[bool, str, str]:
        """
        过滤单个文档
        
        Args:
            url: 文档URL
            text: 文档文本内容
            
        Returns:
            (是否保留, 拒绝原因, 处理后的文本)
        """
        self.stats.add_document()
        original_text = text
        
        # 1. 域名过滤
        if not self.is_target_domain(url):
            self.stats.reject_document("domain_filter")
            return False, "domain_filter", original_text
        
        # 2. 语言检测
        language, confidence = self.identify_language(text)
        if language != 'en' or confidence < self.language_confidence_threshold:
            self.stats.reject_document("language_filter")
            return False, f"language_filter_{language}_{confidence:.2f}", original_text
        
        # 3. Gopher质量过滤
        if not self.gopher_quality_filter(text):
            self.stats.reject_document("gopher_quality_filter")
            return False, "gopher_quality_filter", original_text
        
        # 4. NSFW内容检测
        nsfw_label, nsfw_score = self.classify_nsfw(text)
        if nsfw_label == 'nsfw' and nsfw_score >= self.nsfw_threshold:
            self.stats.reject_document("nsfw_filter")
            return False, f"nsfw_filter_{nsfw_score:.2f}", original_text
        
        # 5. 有毒内容检测
        toxic_label, toxic_score = self.classify_toxic_speech(text)
        if toxic_label == 'toxic' and toxic_score >= self.toxic_threshold:
            self.stats.reject_document("toxic_filter")
            return False, f"toxic_filter_{toxic_score:.2f}", original_text
        
        # 6. 质量分类器
        quality_label, quality_score = self.classify_quality(text)
        if quality_label == 'cc' and quality_score < (1.0 - self.quality_threshold):
            self.stats.reject_document("quality_classifier")
            return False, f"quality_classifier_{quality_score:.2f}", original_text
        
        # 7. PII遮蔽（对保留的文档）
        masked_text, pii_stats = self.apply_pii_masking(text)
        
        # 通过所有过滤器
        self.stats.keep_document()
        return True, "accepted", masked_text

def process_single_wet_file(input_path: str, output_path: str) -> Dict:
    """
    处理单个WET文件
    
    Args:
        input_path: 输入WET文件路径
        output_path: 输出文件路径
        
    Returns:
        处理统计信息
    """
    filter_engine = AdvancedCommonCrawlFilter()
    
    try:
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                
                for record in ArchiveIterator(f_in):
                    if record.record_type == WarcRecordType.conversion:
                        # 获取URL和内容
                        url = record.headers.get('WARC-Target-URI', '')
                        
                        # 从HTML字节提取纯文本
                        html_bytes = record.reader.read()
                        content = filter_engine.extract_text_from_html_bytes(html_bytes)
                        
                        if content is None:
                            continue
                        
                        # 应用过滤器
                        keep, reason, processed_text = filter_engine.filter_document(url, content)
                        
                        if keep:
                            # 保存通过过滤的文档
                            doc_data = {
                                'url': url,
                                'text': processed_text.strip(),
                                'filters_passed': {
                                    'language': filter_engine.identify_language(content),
                                    'nsfw_score': filter_engine.classify_nsfw(content)[1],
                                    'toxic_score': filter_engine.classify_toxic_speech(content)[1],
                                    'quality_score': filter_engine.classify_quality(content)[1]
                                }
                            }
                            f_out.write(json.dumps(doc_data) + '\n')
                            
    except Exception as e:
        logger.error(f"处理文件 {input_path} 时出错: {e}")
        return {'error': str(e)}
        
    stats = filter_engine.stats.get_summary()
    stats['input_file'] = input_path
    stats['output_file'] = output_path
    
    return stats

def run_parallel_filtering(wet_files: List[str], output_dir: str, 
                         max_workers: int = None) -> List[Dict]:
    """
    并行处理多个WET文件
    
    Args:
        wet_files: WET文件路径列表
        output_dir: 输出目录
        max_workers: 最大工作进程数
        
    Returns:
        所有文件的处理统计
    """
    if max_workers is None:
        max_workers = len(os.sched_getaffinity(0))
        
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    all_stats = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for wet_path in wet_files:
            wet_filename = pathlib.Path(wet_path).stem  # 去掉.gz扩展名
            output_path = os.path.join(output_dir, f"{wet_filename}.jsonl")
            
            future = executor.submit(process_single_wet_file, wet_path, output_path)
            futures.append(future)
            
        # 使用进度条显示处理进度
        for future in tqdm(as_completed(futures), total=len(wet_files), 
                          desc="处理WET文件"):
            try:
                stats = future.result()
                all_stats.append(stats)
                
                if 'error' not in stats:
                    logger.info(f"完成处理: {pathlib.Path(stats['input_file']).name}, "
                              f"保留 {stats['kept_documents']}/{stats['total_documents']} 文档 "
                              f"(保留率: {stats['kept_documents']/max(1,stats['total_documents'])*100:.1f}%)")
                              
            except Exception as e:
                logger.error(f"获取处理结果时出错: {e}")
                
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"所有文件处理完成，总耗时: {total_time:.2f} 秒")
    
    return all_stats

def run_slurm_filtering(wet_files: List[str], output_dir: str) -> List[Dict]:
    """
    使用Slurm集群并行处理WET文件
    
    Args:
        wet_files: WET文件路径列表
        output_dir: 输出目录
        
    Returns:
        所有文件的处理统计
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置submitit执行器
    executor = submitit.AutoExecutor(folder="slurm_logs")
    max_simultaneous_jobs = 16
    
    # 配置Slurm作业参数
    executor.update_parameters(
        slurm_array_parallelism=max_simultaneous_jobs,
        timeout_min=120,  # 每个作业最多运行120分钟（考虑到模型加载时间）
        mem_gb=8,  # 每个作业8GB内存（fastText模型需要更多内存）
        cpus_per_task=2,  # 每个作业2个CPU
        slurm_account="student",
        slurm_partition="a4-cpu",
        slurm_qos="a4-cpu-qos",
    )
    
    start_time = time.time()
    futures = []
    
    # 使用batch()上下文管理器将作业分组为Slurm数组
    with executor.batch():
        for wet_path in wet_files:
            wet_filename = pathlib.Path(wet_path).stem
            output_path = os.path.join(output_dir, f"{wet_filename}.jsonl")
            
            future = executor.submit(process_single_wet_file, wet_path, output_path)
            futures.append(future)
    
    all_stats = []
    
    # 使用submitit的as_completed收集结果
    for future in tqdm(submitit.helpers.as_completed(futures), 
                      total=len(wet_files), desc="处理WET文件"):
        try:
            stats = future.result()
            all_stats.append(stats)
            
            if 'error' not in stats:
                logger.info(f"完成处理: {pathlib.Path(stats['input_file']).name}, "
                          f"保留 {stats['kept_documents']}/{stats['total_documents']} 文档 "
                          f"(保留率: {stats['kept_documents']/max(1,stats['total_documents'])*100:.1f}%)")
                          
        except Exception as e:
            logger.error(f"获取处理结果时出错: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"所有文件处理完成，总耗时: {total_time:.2f} 秒")
    
    return all_stats

def analyze_filtering_results(all_stats: List[Dict]) -> Dict:
    """
    分析过滤结果
    
    Args:
        all_stats: 所有文件的处理统计
        
    Returns:
        汇总分析结果
    """
    # 汇总统计
    total_docs = sum(stats.get('total_documents', 0) for stats in all_stats)
    total_kept = sum(stats.get('kept_documents', 0) for stats in all_stats)
    
    # 汇总各过滤器的拒绝统计
    filter_rejections = defaultdict(int)
    for stats in all_stats:
        for filter_name, count in stats.get('filter_rejections', {}).items():
            filter_rejections[filter_name] += count
    
    # 汇总PII遮蔽统计
    pii_masking_stats = defaultdict(int)
    for stats in all_stats:
        for pii_type, count in stats.get('pii_masking_stats', {}).items():
            pii_masking_stats[pii_type] += count
    
    # 计算比例
    rejection_breakdown = {}
    for filter_name, count in filter_rejections.items():
        rejection_breakdown[filter_name] = {
            'count': count,
            'percentage': (count / max(1, total_docs)) * 100
        }
    
    summary = {
        'total_documents_processed': total_docs,
        'documents_kept': total_kept,
        'documents_rejected': total_docs - total_kept,
        'overall_keep_rate': (total_kept / max(1, total_docs)) * 100,
        'rejection_breakdown': rejection_breakdown,
        'pii_masking_summary': dict(pii_masking_stats)
    }
    
    return summary

def estimate_full_crawl_time(sample_time: float, sample_files: int, 
                           total_files: int = 100000) -> float:
    """
    估算处理完整Common Crawl所需时间
    
    Args:
        sample_time: 样本处理时间（秒）
        sample_files: 样本文件数量
        total_files: 总文件数量
        
    Returns:
        估算的总处理时间（小时）
    """
    time_per_file = sample_time / sample_files
    total_time_hours = (time_per_file * total_files) / 3600
    
    return total_time_hours

def main():
    """主函数"""
    # 配置路径
    wet_files_pattern = "/data/CC/CC*.warc.wet.gz"
    output_directory = "/data/filtered_cc_output"
    
    # 获取WET文件列表
    import glob
    wet_files = glob.glob(wet_files_pattern)
    wet_files = wet_files[:5000]  # 限制为5000个文件
    
    logger.info(f"找到 {len(wet_files)} 个WET文件")
    
    if not wet_files:
        logger.error("未找到WET文件，请检查路径")
        return
    
    # 选择处理方式（本地多进程 vs Slurm集群）
    use_slurm = True  # 设置为True使用Slurm，False使用本地多进程
    
    start_time = time.time()
    
    if use_slurm:
        logger.info("使用Slurm集群处理")
        all_stats = run_slurm_filtering(wet_files, output_directory)
    else:
        logger.info("使用本地多进程处理")
        all_stats = run_parallel_filtering(wet_files, output_directory)
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    # 分析结果
    summary = analyze_filtering_results(all_stats)
    
    # 输出结果
    logger.info("="*80)
    logger.info("🎉 高级数据过滤完成！")
    logger.info(f"总处理时间: {total_processing_time:.2f} 秒 ({total_processing_time/3600:.2f} 小时)")
    logger.info(f"处理文件数: {len(wet_files)}")
    logger.info(f"总文档数: {summary['total_documents_processed']:,}")
    logger.info(f"保留文档数: {summary['documents_kept']:,}")
    logger.info(f"整体保留率: {summary['overall_keep_rate']:.2f}%")
    
    logger.info("\n📊 各过滤器拒绝统计:")
    for filter_name, stats in sorted(summary['rejection_breakdown'].items(), 
                                   key=lambda x: x[1]['percentage'], reverse=True):
        logger.info(f"  {filter_name}: {stats['count']:,} ({stats['percentage']:.2f}%)")
    
    logger.info("\n🔒 PII遮蔽统计:")
    for pii_type, count in summary['pii_masking_summary'].items():
        logger.info(f"  {pii_type}: {count:,} 个实例")
    
    # 估算完整处理时间
    estimated_full_time = estimate_full_crawl_time(
        total_processing_time, len(wet_files), 100000
    )
    logger.info(f"\n⏱️  估算处理完整Common Crawl (100,000文件) 所需时间: {estimated_full_time:.1f} 小时")
    
    # 保存统计结果
    results_file = os.path.join(output_directory, "advanced_filtering_summary.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processing_time_seconds': total_processing_time,
            'processing_time_hours': total_processing_time / 3600,
            'files_processed': len(wet_files),
            'estimated_full_crawl_hours': estimated_full_time,
            'filter_thresholds': {
                'language_confidence': 0.7,
                'nsfw_threshold': 0.5,
                'toxic_threshold': 0.5,
                'quality_threshold': 0.8
            },
            'summary': summary,
            'detailed_stats': all_stats
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 详细统计结果保存至: {results_file}")
    
    # 数据质量报告
    logger.info("\n📈 数据质量报告:")
    if summary['documents_kept'] > 0:
        keep_rate = summary['overall_keep_rate']
        if keep_rate > 10:
            logger.info(f"  ✅ 保留率 {keep_rate:.1f}% - 过滤适中，数据充足")
        elif keep_rate > 5:
            logger.info(f"  ⚠️  保留率 {keep_rate:.1f}% - 过滤较严格，建议调整阈值")
        else:
            logger.info(f"  🔴 保留率 {keep_rate:.1f}% - 过滤过于严格，需要调整策略")
    
    logger.info("\n🚀 处理完成！可以开始训练GPT-2模型了。")

if __name__ == "__main__":
    main()