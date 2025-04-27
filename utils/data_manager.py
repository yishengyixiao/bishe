import os
import json
import pandas as pd
import logging
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        """初始化数据管理器"""
        self.data_dir = config.get("DATA_DIR")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 历史记录文件路径
        self.history_file = os.path.join(self.data_dir, "history.csv")
        
        # 统计数据文件路径
        self.stats_file = os.path.join(self.data_dir, "stats.json")
        
        # 初始化统计数据
        self.stats = self._load_stats()
    
    def save_verification_result(self, result):
        """保存验证结果到历史记录"""
        try:
            # 添加时间戳
            result['timestamp'] = datetime.now().isoformat()
            
            # 添加调试日志
            logger.info(f"尝试保存验证结果: {result.get('question', '')[:30]}...")
            logger.info(f"历史文件路径: {self.history_file}")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # 检查是否为首次保存
            is_new_file = not os.path.exists(self.history_file)
            
            # 将结果转换为DataFrame
            df = pd.DataFrame([result])
            
            # 确保某些必要字段存在
            if 'similarity' not in df.columns:
                df['similarity'] = 0.0
            if 'is_risk' not in df.columns:
                df['is_risk'] = True
            
            try:
                # 保存到CSV文件
                df.to_csv(self.history_file, mode="a", header=is_new_file, index=False, encoding='utf-8')
                logger.info(f"成功保存到历史文件: {self.history_file}")
                
                # 验证文件是否成功写入
                if os.path.exists(self.history_file):
                    logger.info(f"历史文件存在，大小: {os.path.getsize(self.history_file)} 字节")
                else:
                    logger.error("历史文件保存后不存在！")
                    
                # 更新统计数据
                self._update_stats(result)
                
                return True
            except Exception as e:
                logger.error(f"写入CSV文件失败: {str(e)}")
                
                # 尝试使用备用方法
                try:
                    # 备用方法：直接追加到文本文件
                    with open(self.history_file + ".backup.txt", 'a', encoding='utf-8') as f:
                        f.write(f"{json.dumps(result, ensure_ascii=False)}\n")
                    logger.info("已使用备用方法保存历史记录")
                    return True
                except Exception as e2:
                    logger.error(f"备用保存方法也失败: {str(e2)}")
                    return False
        except Exception as e:
            logger.error(f"保存验证结果失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_history(self, limit=None, filter_func=None):
        """获取历史记录"""
        try:
            logger.info(f"尝试获取历史记录，限制：{limit}")
            
            # 检查历史文件是否存在
            if not os.path.exists(self.history_file):
                logger.warning(f"历史记录文件不存在: {self.history_file}")
                return []
            
            # 确保文件不为空且可读
            if os.path.getsize(self.history_file) == 0:
                logger.warning("历史记录文件为空")
                return []
            
            try:
                # 使用pandas读取CSV，注意引号处理
                df = pd.read_csv(self.history_file, quoting=1, encoding='utf-8')
                logger.info(f"成功读取历史文件，记录数：{len(df)}")
                
                # 如果DataFrame为空
                if df.empty:
                    logger.warning("历史记录DataFrame为空")
                    return []
                    
                # 应用过滤器
                if filter_func:
                    df = df[df.apply(filter_func, axis=1)]
                
                # 按时间戳排序（如果存在）
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp', ascending=False)
                
                # 限制返回数量
                if limit:
                    df = df.head(limit)
                
                # 转换为字典列表
                records = df.to_dict('records')
                logger.info(f"返回 {len(records)} 条历史记录")
                return records
            except Exception as e:
                logger.error(f"读取CSV文件失败: {str(e)}")
                
                # 尝试使用Python内置的csv模块
                import csv
                try:
                    with open(self.history_file, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        records = list(reader)
                    
                    # 限制记录数
                    if limit:
                        records = records[:limit]
                        
                    logger.info(f"使用csv模块读取成功，记录数：{len(records)}")
                    return records
                except Exception as e2:
                    logger.error(f"使用csv模块读取也失败: {str(e2)}")
                    return []
        except Exception as e:
            logger.error(f"获取历史记录失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stats(self):
        """获取统计数据"""
        try:
            # 检查历史文件是否存在
            if not os.path.exists(self.history_file):
                logger.warning(f"历史记录文件不存在: {self.history_file}")
                return {
                    "total_queries": 0,
                    "high_risk_queries": 0,
                    "low_risk_queries": 0,
                    "average_similarity": 0.0,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # 尝试读取CSV文件
            try:
                if os.path.getsize(self.history_file) == 0:
                    logger.warning("历史记录文件为空")
                    return {
                        "total_queries": 0,
                        "high_risk_queries": 0,
                        "low_risk_queries": 0,
                        "average_similarity": 0.0,
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                df = pd.read_csv(self.history_file)
            except Exception as e:
                logger.error(f"读取历史记录失败: {str(e)}")
                return {
                    "total_queries": 0,
                    "high_risk_queries": 0,
                    "low_risk_queries": 0,
                    "average_similarity": 0.0,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(e)
                }
            
            # 计算统计数据
            total_queries = len(df)
            
            # 检查必要列是否存在
            if 'is_risk' in df.columns:
                high_risk_queries = df[df['is_risk'] == True].shape[0]
                low_risk_queries = total_queries - high_risk_queries
            else:
                high_risk_queries = 0
                low_risk_queries = total_queries
            
            # 计算平均相似度
            if 'similarity' in df.columns and total_queries > 0:
                # 过滤掉NaN值
                valid_similarities = df['similarity'].dropna()
                if len(valid_similarities) > 0:
                    average_similarity = valid_similarities.mean()
                else:
                    average_similarity = 0.0
            else:
                average_similarity = 0.0
            
            return {
                "total_queries": total_queries,
                "high_risk_queries": high_risk_queries,
                "low_risk_queries": low_risk_queries,
                "average_similarity": average_similarity,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"计算统计数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回空统计，避免整个功能崩溃
            return {
                "total_queries": 0,
                "high_risk_queries": 0,
                "low_risk_queries": 0,
                "average_similarity": 0.0,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
    
    def _load_stats(self):
        """加载统计数据"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 默认统计数据
        return {
            "total_queries": 0,
            "high_risk_queries": 0,
            "low_risk_queries": 0,
            "average_similarity": 0.0,
            "query_distribution": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _update_stats(self, result):
        """更新统计数据"""
        try:
            # 更新总查询次数
            self.stats["total_queries"] += 1
            
            # 更新风险查询次数
            if result.get("is_risk", False):
                self.stats["high_risk_queries"] += 1
            else:
                self.stats["low_risk_queries"] += 1
            
            # 更新平均相似度
            current_avg = self.stats["average_similarity"]
            current_total = self.stats["total_queries"] - 1  # 减去当前查询
            new_similarity = result.get("similarity", 0.0)
            self.stats["average_similarity"] = (current_avg * current_total + new_similarity) / self.stats["total_queries"]
            
            # 更新查询分布
            # 提取查询的小时
            hour = datetime.now().hour
            hour_key = f"{hour:02d}:00"
            if hour_key not in self.stats["query_distribution"]:
                self.stats["query_distribution"][hour_key] = 0
            self.stats["query_distribution"][hour_key] += 1
            
            # 更新时间戳
            self.stats["last_updated"] = datetime.now().isoformat()
            
            # 保存统计数据
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"更新统计数据失败: {str(e)}")
    
    def generate_report(self, output_file=None):
        """
        生成数据报告
        
        参数:
            output_file: 输出文件路径
        
        返回:
            报告内容
        """
        try:
            # 检查历史记录文件是否存在
            if not os.path.exists(self.history_file):
                return "# 系统报告\n\n没有历史数据可供分析，尚未进行任何查询。"
            
            # 检查文件是否为空
            if os.path.getsize(self.history_file) == 0:
                return "# 系统报告\n\n历史记录文件存在但为空，尚未保存任何查询结果。"
            
            # 尝试读取CSV文件
            try:
                df = pd.read_csv(self.history_file)
            except Exception as e:
                return f"# 系统报告\n\n读取历史记录文件时出错: {str(e)}"
            
            # 检查数据是否为空
            if len(df) == 0:
                return "# 系统报告\n\n历史记录文件中没有数据。"
            
            # 基本统计
            total_queries = len(df)
            
            # 检查必要的列是否存在
            if 'is_risk' not in df.columns:
                high_risk = 0
                low_risk = total_queries
                note = "\n\n注意: 历史记录缺少风险评估数据。"
            else:
                high_risk = df[df['is_risk'] == True].shape[0]
                low_risk = total_queries - high_risk
                note = ""
            
            # 检查相似度列是否存在
            if 'similarity' not in df.columns:
                avg_similarity = 0
                similarity_dist = "历史记录缺少相似度数据。"
            else:
                avg_similarity = df['similarity'].mean()
                # 相似度分布
                similarity_dist = f"""- 0.0-0.2: {df[df['similarity'].between(0, 0.2)].shape[0]} 次
- 0.2-0.4: {df[df['similarity'].between(0.2, 0.4)].shape[0]} 次
- 0.4-0.6: {df[df['similarity'].between(0.4, 0.6)].shape[0]} 次
- 0.6-0.8: {df[df['similarity'].between(0.6, 0.8)].shape[0]} 次
- 0.8-1.0: {df[df['similarity'].between(0.8, 1.0)].shape[0]} 次"""
            
            # 生成报告
            report = f"""# 问答验证系统数据报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 基本统计
- 总查询次数: {total_queries}
- 高风险查询: {high_risk} ({high_risk/total_queries*100:.1f}% 如果有数据)
- 低风险查询: {low_risk} ({low_risk/total_queries*100:.1f}% 如果有数据)
- 平均相似度: {avg_similarity:.4f} (如果有数据)

## 相似度分布
{similarity_dist}
{note}
"""
            
            # 保存报告
            if output_file:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(report)
                    logger.info(f"报告已保存至 {output_file}")
                except Exception as e:
                    logger.error(f"保存报告文件失败: {str(e)}")
            
            return report
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"# 系统报告生成失败\n\n出现错误: {str(e)}\n\n这可能是由于历史记录文件格式不正确或缺少必要数据。"
    
    def clear_history(self, backup=True):
        """
        清除历史记录
        
        参数:
            backup: 是否备份历史记录
        
        返回:
            是否成功
        """
        try:
            if not os.path.exists(self.history_file):
                return True
                
            # 备份历史记录
            if backup:
                backup_file = f"{self.history_file}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
                os.rename(self.history_file, backup_file)
                logger.info(f"历史记录已备份到 {backup_file}")
            else:
                os.remove(self.history_file)
                logger.info("历史记录已删除")
            
            return True
        except Exception as e:
            logger.error(f"清除历史记录失败: {str(e)}")
            return False
