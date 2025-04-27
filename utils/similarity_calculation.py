from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba
import difflib
import logging

logger = logging.getLogger(__name__)

def calculate_similarity(text1, text2, method="tfidf"):
    """
    计算两段文本的相似度
    
    参数:
        text1, text2: 需要比较的两段文本
        method: 相似度计算方法，可选 "tfidf", "count", "jaccard", "levenshtein"
    
    返回:
        相似度值（0到1之间）
    """
    if not text1 or not text2:
        return 0.0
    
    # 打印输入文本，用于调试
    print(f"计算相似度: '{text1}' 和 '{text2}'")
    
    try:
        if method == "tfidf":
            # 对中文文本进行分词
            words1 = ' '.join(jieba.cut(text1))
            words2 = ' '.join(jieba.cut(text2))
            
            # TF-IDF 向量化
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([words1, words2])
            
            # 计算余弦相似度
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            print(f"TF-IDF 相似度: {similarity}")
            return float(similarity)
        
        elif method == "count":
            # 词频向量化
            vectorizer = CountVectorizer()
            count_matrix = vectorizer.fit_transform([text1, text2])
            return float(cosine_similarity(count_matrix[0:1], count_matrix[1:2])[0][0])
        
        elif method == "jaccard":
            # Jaccard 相似度
            words1 = set(jieba.cut(text1))
            words2 = set(jieba.cut(text2))
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            similarity = len(intersection) / len(union) if union else 0.0
            
            print(f"Jaccard 相似度: {similarity}")
            return similarity
        
        elif method == "levenshtein":
            # Levenshtein 距离（编辑距离）
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
            
            print(f"Levenshtein 相似度: {similarity}")
            return similarity
        
        else:
            # 默认使用 TF-IDF
            return calculate_similarity(text1, text2, "tfidf")
            
    except Exception as e:
        print(f"计算相似度时出错: {str(e)}")
        # 尝试使用备用方法
        try:
            import difflib
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
            print(f"备用相似度计算: {similarity}")
            return similarity
        except:
            return 0.0

def calculate_multiple_similarities(text1, text_list, method="tfidf"):
    """
    计算一段文本与多段文本的相似度
    
    参数:
        text1: 基准文本
        text_list: 文本列表
        method: 相似度计算方法
    
    返回:
        相似度列表
    """
    if not text_list:
        return []
        
    return [calculate_similarity(text1, text2, method) for text2 in text_list]

def get_best_match(text, candidates, method="tfidf"):
    """
    获取最佳匹配的文本
    
    参数:
        text: 基准文本
        candidates: 候选文本列表
        method: 相似度计算方法
    
    返回:
        (最佳匹配文本, 相似度)
    """
    if not candidates:
        return None, 0.0
        
    similarities = calculate_multiple_similarities(text, candidates, method)
    best_index = np.argmax(similarities)
    return candidates[best_index], similarities[best_index]

def get_similarity_matrix(texts, method="tfidf"):
    """
    计算文本列表中所有文本之间的相似度矩阵
    
    参数:
        texts: 文本列表
        method: 相似度计算方法
    
    返回:
        相似度矩阵
    """
    n = len(texts)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                sim = calculate_similarity(texts[i], texts[j], method)
                matrix[i][j] = sim
                matrix[j][i] = sim
    
    return matrix 