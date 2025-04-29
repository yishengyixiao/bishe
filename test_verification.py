import unittest
import sys
import os
import logging
from reverse_verify import verify_answer
from utils.knowledge_base import KnowledgeBase

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestVerification(unittest.TestCase):
    """验证系统功能测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化知识库并添加测试用的知识条目
        self.knowledge_base = KnowledgeBase()
        
        # 保存原始知识库
        self.original_kb = self.knowledge_base.get_all_knowledge()
        
        # 清空并重新添加知识条目用于测试
        self.knowledge_base.clear_knowledge()
        
        test_knowledge = {
            "感冒": "感冒是一种常见的呼吸道疾病，通常症状包括咳嗽、流鼻涕、喉咙痛等，多休息和多喝水有助于恢复。",
            "抑郁症": "抑郁症是一种常见的心理健康问题，需要专业的心理治疗和医疗干预。",
            "人工智能": "人工智能是计算机科学的一个领域，研究如何使计算机模拟人类智能行为。",
            "地球": "地球是太阳系中的第三颗行星，是已知唯一有生命存在的天体。",
        }
        
        # 添加测试用知识
        for key, value in test_knowledge.items():
            self.knowledge_base.add_knowledge(key, value)
        
        logger.info("测试环境已设置")
    
    def tearDown(self):
        """测试后的清理工作"""
        # 还原知识库
        self.knowledge_base.clear_knowledge()
        for key, value in self.original_kb.items():
            self.knowledge_base.add_knowledge(key, value)
        logger.info("测试环境已清理")
    
    def test_similarity_calculation(self):
        """测试相似度计算的准确性"""
        logger.info("开始测试相似度计算")
        
        # 测试相似问题 - 期望高相似度
        similar_questions = [
            "感冒了怎么办？",
            "如何治疗感冒？"
        ]
        
        result = verify_answer(similar_questions[0], save_history=False)
        logger.info(f"问题：{similar_questions[0]}")
        logger.info(f"答案：{result['answer']}")
        logger.info(f"反向问题：{result['reversed_question']}")
        logger.info(f"相似度：{result['similarity']}")
        
        # 相似度应该在合理范围内（通常>0.5）
        self.assertGreaterEqual(result['similarity'], 0.5, 
                              "相似问题的相似度应该较高")
        
        # 测试不相关问题 - 期望低相似度
        unrelated_questions = [
            "地球是什么形状？",
            "人工智能的历史是什么？"
        ]
        
        result = verify_answer(unrelated_questions[0], save_history=False)
        logger.info(f"问题：{unrelated_questions[0]}")
        logger.info(f"答案：{result['answer']}")
        logger.info(f"反向问题：{result['reversed_question']}")
        logger.info(f"相似度：{result['similarity']}")
        
        # 验证返回结构完整性
        self.assertIn('risk_score', result, "结果应包含风险评分")
        self.assertIn('is_risk', result, "结果应包含风险判断")
        
        logger.info("相似度计算测试完成")
    
    def test_multidimensional_detection(self):
        """测试多维度检测的协同工作"""
        logger.info("开始测试多维度检测")
        
        # 测试正常答案
        normal_question = "人工智能有哪些应用？"
        result = verify_answer(normal_question, save_history=False)
        
        logger.info(f"问题：{normal_question}")
        logger.info(f"检测方法数量：{len(result['detection_methods'])}")
        
        # 验证检测方法是否全部运行
        self.assertIn('reverse_question', result['detection_methods'], 
                    "应包含反向问题检测方法")
        self.assertIn('topic_drift', result['detection_methods'], 
                    "应包含主题漂移检测方法")
        self.assertIn('factuality', result['detection_methods'], 
                    "应包含事实性检测方法")
        self.assertIn('knowledge_conflict', result['detection_methods'], 
                    "应包含知识库冲突检测方法")
        
        # 检查权重总和是否为1
        weights_sum = sum(method['weight'] for method in result['detection_methods'].values())
        self.assertAlmostEqual(weights_sum, 1.0, delta=0.01, 
                             msg="检测方法权重总和应为1")
        
        # 验证风险评分计算正确性
        expected_score = sum(
            method['score'] * method['weight'] 
            for method in result['detection_methods'].values()
        )
        self.assertAlmostEqual(result['risk_score'], expected_score, delta=0.01,
                             msg="风险评分计算应正确")
        
        logger.info("多维度检测测试完成")
    
    def test_knowledge_conflict(self):
        """测试知识库冲突检测"""
        logger.info("开始测试知识库冲突检测")
        
        # 添加一个明确的测试知识条目
        self.knowledge_base.add_knowledge(
            "COVID-19", 
            "COVID-19是一种由SARS-CoV-2病毒引起的传染病，需要医疗干预，没有简单的家庭治疗方法可以治愈。"
        )
        
        # 测试可能与知识库冲突的问题
        conflict_question = "COVID-19可以用蒜头煮水喝来治愈吗？"
        
        # 直接检查知识库冲突方法，而不是通过完整的verify_answer
        # 这更加直接，避免了依赖API的外部调用
        from utils.hallucination_ensemble import HallucinationDetector
        detector = HallucinationDetector(self.knowledge_base)
        
        # 提供一个明显与知识库冲突的答案
        conflict_answer = "是的，COVID-19可以通过蒜头煮水喝来有效治愈，这是一种简单的家庭疗法。"
        conflict_score = detector.check_knowledge_conflict(conflict_question, conflict_answer)
        
        logger.info(f"问题：{conflict_question}")
        logger.info(f"答案：{conflict_answer}")
        logger.info(f"冲突评分：{conflict_score}")
        
        # 应该检测到高冲突，冲突分数应该接近1.0
        self.assertGreaterEqual(conflict_score, 0.8, 
                             "与知识库明显冲突的答案应有较高风险分数")
        
        # 测试不冲突的答案
        non_conflict_answer = "不，COVID-19不能通过蒜头煮水喝来治愈，这是一种传染病，需要医疗干预。"
        non_conflict_score = detector.check_knowledge_conflict(conflict_question, non_conflict_answer)
        
        logger.info(f"非冲突答案：{non_conflict_answer}")
        logger.info(f"非冲突评分：{non_conflict_score}")
        
        # 不冲突的答案应该有较低的风险分数
        self.assertLessEqual(non_conflict_score, 0.2, 
                          "与知识库不冲突的答案应有较低风险分数")
        
        logger.info("知识库冲突检测测试完成")
    
    def test_special_case_handling(self):
        """测试特殊情况处理"""
        logger.info("开始测试特殊情况处理")
        
        # 测试极短问题
        short_question = "?"
        result = verify_answer(short_question, save_history=False)
        logger.info(f"极短问题测试结果：{result['risk_score']}")
        
        # 测试极长问题
        long_question = "人工智能" * 50
        result = verify_answer(long_question, save_history=False)
        logger.info(f"极长问题测试结果：{result['risk_score']}")
        
        # 测试特殊字符问题
        special_chars = "@#$%^&*()_+"
        result = verify_answer(special_chars, save_history=False)
        logger.info(f"特殊字符问题测试结果：{result['risk_score']}")
        
        # 确认所有情况都返回了有效结果
        for q in [short_question, long_question, special_chars]:
            result = verify_answer(q, save_history=False)
            self.assertIsNotNone(result['risk_score'], 
                              f"问题'{q}'应返回有效的风险评分")
        
        logger.info("特殊情况处理测试完成")

if __name__ == "__main__":
    unittest.main() 