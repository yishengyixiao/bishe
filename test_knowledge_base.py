import unittest
import os
import logging
from utils.knowledge_base import KnowledgeBase

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestKnowledgeBase(unittest.TestCase):
    """知识库冲突检测功能测试"""
    
    def setUp(self):
        """测试前准备工作"""
        # 初始化知识库
        self.kb = KnowledgeBase()
        
        # 备份原始知识库
        self.original_kb = self.kb.get_all_knowledge()
        
        # 清空并添加测试数据
        self.kb.clear_knowledge()
        
        # 添加测试知识条目
        test_data = {
            "高血压": "高血压是一种慢性疾病，需要长期药物治疗和生活方式调整，没有简单的方法可以完全治愈。",
            "糖尿病": "糖尿病是一种慢性代谢疾病，需要长期管理血糖水平，1型糖尿病需要胰岛素治疗。",
            "阿司匹林": "阿司匹林是一种常用的非处方药，有抗炎、解热、镇痛和抗血小板聚集的作用，不适合胃溃疡患者。",
            "减肥": "健康减肥需要均衡饮食和适当运动，没有安全的速效方法可以在短时间内大量减重。"
        }
        
        for key, value in test_data.items():
            self.kb.add_knowledge(key, value)
            
        logger.info(f"知识库初始化完成，共 {len(test_data)} 条知识")
    
    def tearDown(self):
        """测试后清理工作"""
        # 恢复原始知识库
        self.kb.clear_knowledge()
        for key, value in self.original_kb.items():
            self.kb.add_knowledge(key, value)
        logger.info("原始知识库已恢复")
    
    def test_add_knowledge(self):
        """测试添加知识功能"""
        # 添加新知识
        success = self.kb.add_knowledge("测试关键词", "这是测试内容")
        self.assertTrue(success, "添加知识应该成功")
        
        # 验证知识是否添加成功
        kb_data = self.kb.get_all_knowledge()
        self.assertIn("测试关键词", kb_data, "新添加的知识应该在知识库中")
        self.assertEqual(kb_data["测试关键词"], "这是测试内容", "知识内容应该匹配")
    
    def test_knowledge_conflict_basic(self):
        """测试基本的知识冲突检测"""
        # 明显冲突的情况
        question = "高血压可以通过喝柠檬水一周内治愈吗？"
        answer = "是的，每天喝柠檬水可以在一周内治愈高血压。"
        
        is_conflict, info = self.kb.check_conflict(question, answer)
        logger.info(f"冲突检测结果: {is_conflict}, 信息: {info}")
        
        self.assertTrue(is_conflict, "应该检测到与知识库冲突")
        self.assertIn("高血压", info, "冲突信息应该包含关键词")
    
    def test_knowledge_conflict_partial(self):
        """测试部分冲突情况"""
        # 部分冲突（答案中有部分正确信息，也有错误信息）
        question = "糖尿病患者应该如何饮食？"
        answer = "糖尿病患者应该控制碳水化合物摄入，多吃蔬菜，但只要坚持两周特殊饮食就能完全治愈糖尿病，不再需要药物。"
        
        is_conflict, info = self.kb.check_conflict(question, answer)
        
        self.assertTrue(is_conflict, "应该检测到与知识库部分冲突")
        self.assertIn("糖尿病", info, "冲突信息应该包含关键词")
    
    def test_knowledge_conflict_negative(self):
        """测试无冲突情况"""
        # 无冲突的情况
        question = "阿司匹林有什么作用？"
        answer = "阿司匹林是一种常用药物，具有解热、镇痛、抗炎和抗血小板聚集的作用。"
        
        is_conflict, info = self.kb.check_conflict(question, answer)
        
        self.assertFalse(is_conflict, "不应该检测到冲突")
        self.assertEqual(info, "", "无冲突时信息应为空")
    
    def test_knowledge_conflict_edge_cases(self):
        """测试边缘情况"""
        # 空问题
        is_conflict, _ = self.kb.check_conflict("", "测试答案")
        self.assertFalse(is_conflict, "空问题不应该检测到冲突")
        
        # 空答案
        is_conflict, _ = self.kb.check_conflict("测试问题", "")
        self.assertFalse(is_conflict, "空答案不应该检测到冲突")
        
        # 问题和答案都为空
        is_conflict, _ = self.kb.check_conflict("", "")
        self.assertFalse(is_conflict, "空问题和空答案不应该检测到冲突")
    
    def test_knowledge_similarity_threshold(self):
        """测试知识相似度阈值"""
        # 添加一个特定的知识条目
        self.kb.add_knowledge("感冒", "感冒是一种常见疾病，通常会自行痊愈，但需要休息和多喝水。")
        
        # 测试轻微变化的问题
        slight_variants = [
            "感冒怎么办？",
            "我感冒了怎么治疗？",
            "如何处理感冒症状？",
            "感冒了需要吃药吗？"
        ]
        
        # 不同程度冲突的答案
        answers = [
            # 应该检测到冲突的答案
            "感冒可以通过特定草药完全治愈，无需休息。",
            # 可能检测到冲突的答案
            "感冒需要使用强效抗生素立即治疗，否则会发展成肺炎。",
            # 不应该检测到冲突的答案
            "感冒通常需要休息和多喝水，一般会在一周内自行痊愈。"
        ]
        
        # 测试不同问题和答案组合
        results = []
        for question in slight_variants:
            for answer in answers:
                is_conflict, info = self.kb.check_conflict(question, answer)
                results.append((question, answer, is_conflict))
                logger.info(f"问题: {question}, 答案: {answer}, 冲突: {is_conflict}")
        
        # 检查是否有检测到的冲突
        conflict_count = sum(1 for _, _, is_conflict in results if is_conflict)
        logger.info(f"测试了 {len(results)} 种组合，检测到 {conflict_count} 个冲突")
        
        # 应该至少检测到一些冲突
        self.assertGreater(conflict_count, 0, "应该至少检测到一些知识冲突")

if __name__ == "__main__":
    unittest.main() 