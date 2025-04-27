import logging
from utils.similarity_calculation import calculate_similarity
import re
import numpy as np
from config import config

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """集成多种方法检测AI回答中的幻觉"""
    
    def __init__(self, knowledge_base):
        """初始化检测器"""
        self.knowledge_base = knowledge_base
        # 调整权重，增加反向问题的权重
        self.weights = {
            "reverse_question": 0.6,  # 从0.4提高到0.6
            "topic_drift": 0.15,     # 从0.2降低到0.15
            "factuality": 0.15,      # 从0.3降低到0.15
            "knowledge_conflict": 0.1  # 保持不变
        }
    
    def detect_with_all_methods(self, question, answer, original_similarity=None, factuality_score=None):
        """使用所有方法检测幻觉"""
        results = {}
        
        # 1. 反向问题法 - 相似度越低，风险越高
        if original_similarity is not None:
            reverse_score = 1.0 - original_similarity  # 相似度0.5转为风险分数0.5
            results["reverse_question"] = {
                "score": reverse_score,
                "weight": self.weights["reverse_question"],  # 权重0.6
                "details": "基于反向问题与原问题的相似度"
            }
        
        # 2. 主题漂移检测
        topic_drift_score = self.detect_topic_drift(question, answer)
        results["topic_drift"] = {
            "score": topic_drift_score,
            "weight": self.weights["topic_drift"],
            "details": "检测答案是否偏离了问题的主题"
        }
        
        # 3. 事实性评分
        if factuality_score is not None:
            # 将1-10的分数转换为0-1之间的风险分数 (分数越高风险越低)
            fact_risk_score = 1.0 - (factuality_score / 10.0)
            results["factuality"] = {
                "score": fact_risk_score,
                "weight": self.weights["factuality"],
                "details": "基于事实性评分的风险值"
            }
        
        # 4. 知识库冲突检测
        conflict_score = self.check_knowledge_conflict(question, answer)
        results["knowledge_conflict"] = {
            "score": conflict_score,
            "weight": self.weights["knowledge_conflict"],
            "details": "检测答案是否与知识库冲突"
        }
        
        # 计算加权风险分数
        total_weight = sum(method["weight"] for method in results.values())
        if total_weight == 0:
            total_weight = 1  # 防止除零错误
            
        weighted_score = sum(method["score"] * method["weight"] for method in results.values()) / total_weight
        
        # 修改判断逻辑，降低风险阈值
        # 确定是否存在幻觉 (风险分数>阈值视为有幻觉)
        is_hallucination = weighted_score > float(config.get("RISK_THRESHOLD", 0.5))
        
        # 添加特殊规则：如果相似度过低，直接判定为高风险
        if original_similarity is not None and original_similarity < float(config.get("SIMILARITY_CRITICAL", 0.4)):
            is_hallucination = True
        
        return {
            "is_hallucination": is_hallucination,
            "risk_score": weighted_score,
            "methods": results
        }
    
    def detect_topic_drift(self, question, answer):
        """检测主题漂移 - 答案是否偏离了问题的主题"""
        try:
            # 提取问题中的关键词
            question_keywords = set(re.findall(r'\w+', question.lower()))
            
            # 检查这些关键词在答案中的覆盖率
            keywords_found = 0
            for keyword in question_keywords:
                if len(keyword) > 2 and keyword in answer.lower():
                    keywords_found += 1
            
            # 计算匹配率
            if len(question_keywords) > 0:
                coverage = keywords_found / len(question_keywords)
                # 转换为风险分数 (覆盖率越低，风险越高)
                risk_score = 1.0 - coverage
                return min(1.0, max(0.0, risk_score))
            
            return 0.5  # 默认适中风险
        except Exception as e:
            logger.error(f"主题漂移检测出错: {str(e)}")
            return 0.5  # 出错时返回中等风险值
    
    def check_knowledge_conflict(self, question, answer):
        """检查答案是否与知识库冲突"""
        try:
            is_conflict, _ = self.knowledge_base.check_conflict(question, answer)
            return 0.9 if is_conflict else 0.1  # 有冲突时返回高风险分数
        except Exception as e:
            logger.error(f"知识库冲突检测出错: {str(e)}")
            return 0.5  # 出错时返回中等风险值 