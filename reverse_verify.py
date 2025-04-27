import os
import logging
import pandas as pd
import numpy as np
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from utils.keyword_extraction import extract_keywords, analyze_text
from utils.similarity_calculation import calculate_similarity, calculate_multiple_similarities
from utils.api_interaction import generate_text, generate_multiple_responses, check_factuality
from utils.knowledge_base import KnowledgeBase
from utils.data_manager import DataManager
from config import config
from utils.hallucination_ensemble import HallucinationDetector

# 配置日志记录
log_dir = config.get("LOG_DIR")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "verification.log"),
    level=getattr(logging, config.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reverse_verify")

# 添加终端处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# 加载环境变量
load_dotenv()

# 初始化知识库
knowledge_base = KnowledgeBase()

# 初始化数据管理器
data_manager = DataManager()

# 初始化检测器
hallucination_detector = HallucinationDetector(knowledge_base)

def reverse_questions(original_question, answer, generate_times=3):
    """
    根据答案反向生成问题，并选择与原问题最相似的一个
    
    参数:
        original_question: 用户的原始问题
        answer: 生成的答案
        generate_times: 生成反向问题的次数
    
    返回:
        最相似的问题、相似度和所有生成的问题
    """
    start_time = time.time()
    
    # 修改这里，限制关键词数量为3个
    keywords = extract_keywords(original_question, topK=3, min_keywords=2)
    # 只取前2-3个最重要的关键词
    keywords = keywords[:min(3, len(keywords))]
    
    reversed_questions = []
    
    # 记录开始生成反向问题
    logger.info(f"开始生成反向问题 - 原问题: '{original_question}', 关键词: {keywords}")
    
    for i in range(generate_times):
        # 根据迭代次数调整提示模板
        if i == 0:
            prompt = f"""请根据答案生成对应的原始问题，要求：
必须包含以下1-2个关键词：{", ".join(keywords[:2])}
问题长度不超过15字
保持口语化表达

答案：{answer}
生成的问题："""
        else:
            prompt = f"""请根据答案生成一个不同的原始问题，要求：
必须包含至少1个关键词：{", ".join(keywords[:2])}
问题长度不超过15字
保持口语化表达
与之前生成的问题不同

答案：{answer}
已生成的问题：{', '.join(reversed_questions)}
新生成的问题："""
        
        question = generate_text(prompt, temperature=0.8 + (i * 0.1))
        if question:
            reversed_questions.append(question)
    
    if not reversed_questions:
        logger.warning(f"反向生成问题失败 - 原问题: '{original_question}'")
        return None, 0.0, []
    
    # 计算每个反向问题与原问题的相似度
    similarities = calculate_multiple_similarities(original_question, reversed_questions)
    best_index = np.argmax(similarities)
    
    elapsed_time = time.time() - start_time
    logger.info(f"反向问题生成完成 - 耗时: {elapsed_time:.2f}秒, 最佳问题: '{reversed_questions[best_index]}', 相似度: {similarities[best_index]:.4f}")
    
    return reversed_questions[best_index], similarities[best_index], reversed_questions

def verify_answer(question, save_history=True):
    """
    主验证流程：生成答案，反向生成问题，验证相似度
    
    参数:
        question: 用户输入的问题
        save_history: 是否保存历史记录
    
    返回:
        包含答案、反向问题、相似度和风险评估的字典
    """
    start_time = time.time()
    logger.info(f"开始验证问题: '{question}'")
    print(f"\n========== 开始验证问题 ==========\n{question}")
    
    try:
        # 分析问题文本
        text_analysis = analyze_text(question)
        logger.info(f"问题分析 - 关键词: {text_analysis['keywords']}, 实体: {text_analysis['entities']}")
        
        # 生成答案
        answer = generate_text(f"问题：{question}\n答案：", temperature=0.7)
        print(f"\n响应时间: {time.time() - start_time:.2f}秒")
        
        # 即使答案生成失败，也创建一个基本结果以便记录历史
        if not answer:
            logger.error(f"生成答案失败 - 问题: '{question}'")
            answer = "生成答案时出现问题"
            
            # 创建基本结果
            basic_result = {
                "question": question,
                "answer": answer,
                "reversed_question": "",
                "similarity": 0.0,
                "is_risk": True,
                "error": "生成答案失败",
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存基本历史记录
            if save_history:
                data_manager.save_verification_result(basic_result)
                
            return basic_result
        
        # 检查答案与知识库是否冲突
        is_conflict, conflict_info = knowledge_base.check_conflict(question, answer)
        if is_conflict:
            logger.warning(f"答案与知识库冲突 - {conflict_info}")
        
        # 检查答案的事实性
        factuality_score, factuality_reason = check_factuality(answer)
        logger.info(f"答案事实性评分: {factuality_score}/10, 理由: {factuality_reason}")
        
        # 反向生成问题
        reversed_question, similarity, all_reversed_questions = reverse_questions(question, answer)
        print(f"反向问题: {reversed_question}")
        print(f"相似度: {similarity:.4f}")
        
        # 在计算完相似度后，添加使用集成检测器
        # 从配置中获取最新的阈值
        threshold = float(config.get("SIMILARITY_THRESHOLD"))
        
        # 使用集成检测器
        enhanced_result = hallucination_detector.detect_with_all_methods(
            question, 
            answer,
            original_similarity=similarity,
            factuality_score=factuality_score
        )
        
        # 获取检测器结果
        is_risk = enhanced_result["is_hallucination"]
        risk_score = enhanced_result["risk_score"]
        detection_methods = enhanced_result["methods"]
        
        # 组装结果 - 添加新的检测信息
        result = {
            "question": question,
            "answer": answer,
            "reversed_question": reversed_question,
            "all_reversed_questions": all_reversed_questions,
            "similarity": similarity,
            "is_risk": is_risk,
            "risk_score": risk_score,  # 新增
            "threshold": threshold,
            "factuality_score": factuality_score,
            "factuality_reason": factuality_reason,
            "is_knowledge_conflict": is_conflict,
            "knowledge_conflict_info": conflict_info if is_conflict else "",
            "keywords": text_analysis["keywords"],
            "entities": text_analysis["entities"],
            "detection_methods": detection_methods,  # 新增
            "processing_time": time.time() - start_time
        }
        
        # 记录验证结果
        risk_status = "高风险" if is_risk else "低风险"
        logger.info(f"验证完成 - 问题: '{question}', 相似度: {similarity:.4f}, 风险: {risk_status}, 耗时: {result['processing_time']:.2f}秒")
        
        # 添加打印验证结果总结
        print("\n========== 验证结果 ==========")
        print(f"风险评估: {'高风险' if is_risk else '低风险'} (分数: {risk_score:.4f}, 阈值: {threshold:.2f})")
        print(f"事实性评分: {factuality_score}/10")
        print(f"处理总时间: {result['processing_time']:.2f}秒")
        print("=============================\n")
        
        # 保存历史记录
        if save_history:
            try:
                success = data_manager.save_verification_result(result)
                if not success:
                    logger.error(f"保存验证结果失败 - 问题: '{question}'")
            except Exception as e:
                logger.error(f"保存历史记录时出错: {str(e)}")
        
        return result
    except Exception as e:
        logger.error(f"验证过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"验证过程中出错: {str(e)}"}

def batch_verify(questions):
    """
    批量验证问题
    
    参数:
        questions: 问题列表
    
    返回:
        验证结果列表
    """
    results = []
    for question in questions:
        result = verify_answer(question)
        results.append(result)
    return results

def get_verification_stats():
    """获取验证统计数据"""
    return data_manager.get_stats()

def generate_verification_report(output_file=None):
    """生成验证报告"""
    return data_manager.generate_report(output_file) 