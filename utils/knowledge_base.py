import os
import json
import logging
import pandas as pd
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, file_path=None):
        """
        初始化知识库
        
        参数:
            file_path: 知识库文件路径，如果为 None 则使用默认知识库
        """
        # 默认知识库
        self.knowledge = {
            "癌症": "目前没有简单的治愈方法，需要专业医疗团队制定综合治疗方案",
            "新冠": "需要遵循医生建议和官方指南，注意个人防护和公共卫生",
            "减肥": "健康减肥需要均衡饮食和适当运动，避免极端节食",
            "投资": "投资有风险，需谨慎决策，分散投资降低风险",
            "疫苗": "疫苗接种应遵循医生建议，了解可能的副作用",
            "药物": "用药应遵医嘱，不可自行调整剂量，注意药物相互作用",
            "心理健康": "心理问题需要专业帮助，及时寻求心理医生或咨询师支持",
            "自然疗法": "许多自然疗法缺乏科学验证，不应替代常规医疗",
            "饮食补充剂": "补充剂不受严格监管，效果和安全性可能缺乏充分证据"
        }
        
        # 如果提供了文件路径，从文件加载知识
        if file_path:
            self.load_from_file(file_path)
        else:
            # 尝试从默认位置加载
            default_path = os.path.join(config.get("DATA_DIR"), "knowledge.json")
            if os.path.exists(default_path):
                self.load_from_file(default_path)
    
    def load_from_file(self, file_path=None):
        """从文件加载知识库"""
        if file_path is None:
            file_path = os.path.join(config.get("DATA_DIR"), "knowledge.json")
        
        try:
            # 确保文件存在
            if not os.path.exists(file_path):
                logger.warning(f"知识库文件不存在: {file_path}")
                return False
            
            # 检查文件大小
            if os.path.getsize(file_path) == 0:
                logger.warning(f"知识库文件为空: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 清除旧数据
                self.knowledge.clear()
                
                # 加载新数据
                self.knowledge.update(data)
                
            logger.info(f"从 {file_path} 加载了 {len(data)} 条知识")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"知识库文件格式错误: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"加载知识库失败: {str(e)}")
            return False
    
    def save_to_file(self, file_path=None):
        """保存知识库到文件"""
        if not file_path:
            file_path = os.path.join(config.get("DATA_DIR"), "knowledge.json")
            
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
            logger.info(f"知识库已保存到 {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存知识库失败: {str(e)}")
            return False
    
    def add_knowledge(self, key, value):
        """添加知识"""
        self.knowledge[key] = value
        logger.info(f"添加知识: {key} - {value}")
        return True
    
    def remove_knowledge(self, key):
        """删除知识"""
        if key in self.knowledge:
            del self.knowledge[key]
            logger.info(f"删除知识: {key}")
            return True
        return False
    
    def get_knowledge(self, key):
        """获取知识"""
        return self.knowledge.get(key)
    
    def get_all_knowledge(self):
        """获取所有知识"""
        return self.knowledge
    
    def check_conflict(self, question, answer):
        """
        检查答案是否与知识库冲突
        
        参数:
            question: 用户问题
            answer: 生成的答案
        
        返回:
            (is_conflict, conflict_info): 是否冲突及冲突信息
        """
        for key, value in self.knowledge.items():
            if key in question:
                # 如果问题包含知识库中的关键词，检查答案是否与知识库冲突
                if value not in answer and not any(synonym in answer for synonym in self._get_synonyms(value)):
                    return True, f"答案可能与已知信息冲突: {key} - {value}"
        return False, ""
    
    def _get_synonyms(self, value):
        """获取同义表达（简化版）"""
        # 这里可以扩展为使用同义词库或词向量模型
        return [value]
    
    def export_to_csv(self, file_path=None):
        """导出知识库到CSV文件"""
        if not file_path:
            file_path = os.path.join(config.get("DATA_DIR"), "knowledge_export.csv")
            
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 创建DataFrame并保存
            df = pd.DataFrame(list(self.knowledge.items()), columns=['Key', 'Value'])
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"知识库已导出到 {file_path}，共 {len(df)} 条记录")
            return True
        except Exception as e:
            logger.error(f"导出知识库失败: {str(e)}")
            return False
    
    def import_from_csv(self, file_path):
        """从CSV文件导入知识库"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 尝试不同的编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"成功以 {encoding} 编码读取文件")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error("无法解码文件，尝试了多种编码方式")
                return False
            
            # 检查CSV文件格式
            if 'Key' not in df.columns or 'Value' not in df.columns:
                # 尝试使用第一行和第二行作为Key和Value
                if len(df.columns) >= 2:
                    df.columns = ['Key', 'Value'] + list(df.columns)[2:]
                    logger.info("CSV文件没有正确的列名，已自动重命名前两列为'Key'和'Value'")
                else:
                    logger.error("CSV文件格式错误: 需要'Key'和'Value'列")
                    return False
            
            # 进行导入
            imported = 0
            for _, row in df.iterrows():
                try:
                    key = str(row['Key']).strip()
                    value = str(row['Value']).strip()
                    if key and value and key != 'nan' and value != 'nan':
                        self.knowledge[key] = value
                        imported += 1
                except Exception as e:
                    logger.warning(f"导入行失败: {str(e)}")
                    continue
            
            # 立即保存到文件
            self.save_to_file()
            
            logger.info(f"从 {file_path} 导入了 {imported} 条知识")
            return imported > 0
        except Exception as e:
            logger.error(f"导入知识库失败: {str(e)}")
            return False
