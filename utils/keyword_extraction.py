import jieba
import jieba.analyse
import re
from collections import Counter
import os
import json

# 扩展停用词表
stop_words = set([
    "的", "是", "了", "在", "和", "与", "或", "什么", "怎么", "如何", "为什么",
    "哪些", "这个", "那个", "这些", "那些", "一个", "一些", "有些", "可以", "不能",
    "应该", "需要", "可能", "一定", "必须", "我", "你", "他", "她", "它", "我们", "你们"
])

# 初始化结巴分词
jieba.initialize()

# 添加自定义词典
def add_custom_dictionary(file_path=None, words=None):
    """添加自定义词典，提高分词准确性"""
    if file_path and os.path.exists(file_path):
        jieba.load_userdict(file_path)
        print(f"已加载自定义词典: {file_path}")
    
    if words and isinstance(words, list):
        for word in words:
            jieba.add_word(word)
        print(f"已添加 {len(words)} 个自定义词")

# 加载医学术语词典
medical_terms_path = os.path.join("data", "medical_terms.txt")
if os.path.exists(medical_terms_path):
    add_custom_dictionary(medical_terms_path)

def extract_keywords(question, topK=4, method="tfidf", min_keywords=2):
    """
    提取问题中的关键词
    
    参数:
        question: 用户输入的问题
        topK: 返回的关键词数量
        method: 关键词提取方法，可选 "tfidf" 或 "textrank"
        min_keywords: 最少返回的关键词数量
    
    返回:
        关键词列表
    """
    if not question:
        return []
        
    # 预处理文本
    question = re.sub(r'[^\w\s]', '', question)
    
    if method == "textrank":
        keywords = jieba.analyse.textrank(
            question, 
            topK=topK, 
            withWeight=False,
            allowPOS=('n', 'vn', 'v', 'a', 'an')  # 名词、动名词、动词、形容词
        )
    else:  # 默认使用 TF-IDF
        keywords = jieba.analyse.extract_tags(
            question, 
            topK=topK, 
            withWeight=False,
            allowPOS=('n', 'vn', 'v', 'a', 'an')
        )
    
    # 过滤停用词
    filtered_keywords = [kw for kw in keywords if kw not in stop_words]
    
    # 如果关键词太少，尝试使用词频统计
    if len(filtered_keywords) < min_keywords:
        words = [w for w in jieba.cut(question) if w not in stop_words and len(w) > 1]
        word_counts = Counter(words).most_common(topK)
        filtered_keywords.extend([word for word, _ in word_counts])
        # 去重
        filtered_keywords = list(dict.fromkeys(filtered_keywords))
    
    return filtered_keywords[:topK]

def extract_entities(text):
    """
    提取文本中的实体（人名、地名、机构名等）
    
    参数:
        text: 输入文本
    
    返回:
        实体列表
    """
    words = jieba.posseg.cut(text)
    entities = []
    
    for word, flag in words:
        # nr:人名, ns:地名, nt:机构名, nz:其他专名
        if flag in ['nr', 'ns', 'nt', 'nz'] and len(word) > 1:
            entities.append(word)
    
    return entities

def analyze_text(text):
    """
    全面分析文本，返回关键词、实体和词性分布
    
    参数:
        text: 输入文本
    
    返回:
        分析结果字典
    """
    if not text:
        return {"keywords": [], "entities": [], "pos_stats": {}}
        
    # 提取关键词 - 将topK从8减少到4
    keywords = extract_keywords(text, topK=4)
    
    # 提取实体
    entities = extract_entities(text)
    
    # 词性统计
    words_pos = jieba.posseg.cut(text)
    pos_stats = {}
    for word, flag in words_pos:
        if flag not in pos_stats:
            pos_stats[flag] = 0
        pos_stats[flag] += 1
    
    return {
        "keywords": keywords,
        "entities": entities,
        "pos_stats": pos_stats
    } 