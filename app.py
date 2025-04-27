import gradio as gr
from reverse_verify import verify_answer
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from utils.data_manager import DataManager
from utils.knowledge_base import KnowledgeBase
from config import config
import logging
import shutil
import matplotlib
from matplotlib.font_manager import FontProperties
import csv

# 创建日志目录
logger = logging.getLogger(__name__)

# 创建fonts目录
fonts_dir = "fonts"
os.makedirs(fonts_dir, exist_ok=True)

# 字体文件路径
font_path = os.path.join(fonts_dir, "simhei.ttf")

# 检查字体文件是否存在
if not os.path.exists(font_path):
    # 如果字体文件不存在，尝试使用内置的黑体等效字体
    # 在Windows系统上搜索并复制
    windows_font = r"C:\Windows\Fonts\simhei.ttf"
    if os.path.exists(windows_font):
        try:
            shutil.copy(windows_font, font_path)
            logger.info(f"已从Windows系统复制黑体字体到 {font_path}")
        except Exception as e:
            logger.error(f"复制字体文件失败: {str(e)}")
    else:
        # 在下面的函数中使用英文标签
        logger.warning("无法找到黑体字体文件，将使用英文标签")

# 创建字体对象
if os.path.exists(font_path):
    chinese_font = FontProperties(fname=font_path)
    logger.info("成功加载中文字体文件")
else:
    chinese_font = None
    logger.warning("未加载中文字体文件，将使用系统默认字体")

# 设置中文字体支持
try:
    # 尝试使用微软雅黑字体 (Windows系统常见字体)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
    
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查字体是否成功加载
    matplotlib.font_manager._rebuild()
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 测试中文字体
    test_fig = plt.figure(figsize=(1, 1))
    plt.text(0.5, 0.5, '测试中文', fontsize=9)
    plt.close(test_fig)
    
    logger.info("成功配置中文字体支持")
except Exception as e:
    logger.warning(f"配置中文字体失败: {str(e)}")
    
    # 备用方案：不使用中文
    # 如果无法配置中文字体，将标签改为英文
    CHINESE_TO_ENGLISH = {
        '相似度分布': 'Similarity Distribution',
        '风险分布': 'Risk Distribution',
        '相似度': 'Similarity',
        '频次': 'Frequency',
        '低风险': 'Low Risk',
        '高风险': 'High Risk',
        '风险阈值': 'Risk Threshold',
        '平均值': 'Average',
        '总计': 'Total'
    }

# 初始化数据目录和权限
def initialize_directories():
    data_dir = config.get("DATA_DIR")
    log_dir = config.get("LOG_DIR")
    
    # 确保目录存在并有写入权限
    for dir_path in [data_dir, log_dir]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            # 创建测试文件以验证写入权限
            test_file = os.path.join(dir_path, "permission_test.txt")
            with open(test_file, 'w') as f:
                f.write("Permission test")
            os.remove(test_file)
            logger.info(f"目录 {dir_path} 已创建并可写入")
        except Exception as e:
            logger.error(f"无法创建或写入目录 {dir_path}: {str(e)}")
            # 尝试使用临时目录
            alt_dir = os.path.join(os.path.expanduser("~"), f"bishe_{os.path.basename(dir_path)}")
            os.makedirs(alt_dir, exist_ok=True)
            logger.info(f"将使用替代目录: {alt_dir}")
            if dir_path == data_dir:
                config.set("DATA_DIR", alt_dir)
            else:
                config.set("LOG_DIR", alt_dir)

# 初始化目录
initialize_directories()

# 初始化数据管理器和知识库
data_manager = DataManager()
knowledge_base = KnowledgeBase()

# 知识库管理函数
def add_knowledge_item(key, value):
    """添加或更新知识条目"""
    if not key or not key.strip():
        gr.Warning("关键词不能为空")
        return key, init_kb_table()
    
    success = knowledge_base.add_knowledge(key.strip(), value.strip())
    knowledge_base.save_to_file()
    
    if success:
        gr.Info(f"已添加/更新知识: {key}")
        return "", init_kb_table()
    else:
        gr.Warning("添加/更新知识失败")
        return key, init_kb_table()

def delete_knowledge_item(key):
    """删除知识条目"""
    if not key or not key.strip():
        gr.Warning("请输入要删除的关键词")
        return key, init_kb_table()
    
    success = knowledge_base.remove_knowledge(key.strip())
    knowledge_base.save_to_file()
    
    if success:
        gr.Info(f"已删除知识: {key}")
        return "", init_kb_table()
    else:
        gr.Warning(f"未找到关键词: {key}")
        return key, init_kb_table()

def clear_kb_inputs():
    """清除知识库输入框"""
    return "", ""

def import_knowledge(file_obj):
    """从文件导入知识库"""
    if not file_obj:
        gr.Warning("请选择文件")
        return init_kb_table()
    
    try:
        # 保存上传的文件
        temp_path = file_obj.name
        success = knowledge_base.import_from_csv(temp_path)
        
        if success:
            gr.Info("成功导入知识库")
        else:
            gr.Warning("导入知识库失败")
        
        return init_kb_table()
    except Exception as e:
        gr.Error(f"导入知识库出错: {str(e)}")
        return init_kb_table()

def export_knowledge():
    """导出知识库"""
    try:
        export_path = os.path.join(config.get("DATA_DIR"), "knowledge_export.csv")
        success = knowledge_base.export_to_csv(export_path)
        
        if success:
            return f"知识库已导出到: {export_path}"
        else:
            return "导出知识库失败"
    except Exception as e:
        return f"导出知识库出错: {str(e)}"

# 初始化知识库表格
def init_kb_table():
    kb_data = knowledge_base.get_all_knowledge()
    return [[key, value] for key, value in kb_data.items()]

# 初始化系统统计
def init_system_stats():
    try:
        stats = data_manager.get_stats()
        # 确保返回有效的字典
        if not isinstance(stats, dict):
            logger.warning(f"统计数据格式不正确: {type(stats)}")
            # 提供默认字典
            return {
                "total_queries": 0,
                "high_risk_queries": 0,
                "low_risk_queries": 0,
                "average_similarity": 0.0,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "无法获取有效统计数据"
            }
        return stats
    except Exception as e:
        logger.error(f"获取统计数据失败: {str(e)}")
        # 出错时返回空字典而非None或字符串
        return {
            "error": f"获取统计数据失败: {str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

with gr.Blocks(title="智能问答验证系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧐 智能问答验证系统")
    gr.Markdown("本系统通过反向问题生成技术验证AI回答的可靠性")
    
    with gr.Tabs():
        # 问答验证标签页
        with gr.TabItem("问答验证"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="输入您的问题", 
                        placeholder="例如：感冒了怎么办？", 
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("开始验证", variant="primary")
                        clear_btn = gr.Button("清除")
                
            with gr.Accordion("验证结果", open=True):
                with gr.Row():
                    with gr.Column(scale=3):
                        answer_output = gr.Textbox(label="生成的答案", lines=5)
                        reversed_output = gr.Textbox(label="反向生成的问题", lines=2)
                        
                    with gr.Column(scale=2):
                        with gr.Row():
                            similarity_score = gr.Number(label="相似度评分")
                            factuality_score = gr.Number(label="事实性评分 (1-10)")
                        
                        with gr.Row():
                            risk_indicator = gr.Label(
                                label="风险评估",
                                value={"低风险": 1.0},
                                num_top_classes=2
                            )
                        
                        with gr.Row():
                            risk_score_display = gr.Number(label="综合风险评分")
                        
                        with gr.Row():
                            thresholds_display = gr.Markdown(
                                "风险评分阈值: 0.5 (>0.5为高风险) | 相似度临界值: 0.4 (<0.4直接判为高风险)"
                            )
                        
                        conflict_output = gr.Textbox(label="知识库冲突", visible=True)
            
            with gr.Accordion("详细分析", open=False):
                with gr.Row():
                    with gr.Column():
                        keywords_output = gr.Dataframe(
                            headers=["关键词"],
                            label="提取的关键词"
                        )
                    with gr.Column():
                        all_reversed_output = gr.Dataframe(
                            headers=["反向生成的问题"],
                            label="所有反向生成的问题"
                        )
                
                factuality_reason = gr.Textbox(label="事实性评估理由", lines=2)
                processing_time = gr.Textbox(label="处理时间", value="0秒")
                detection_method_chart = gr.Image(label="检测方法分析")
        
        # 历史记录标签页
        with gr.TabItem("历史记录"):
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_history_btn = gr.Button("刷新历史记录", variant="primary")
                    clear_history_btn = gr.Button("清除历史记录", variant="stop")
                    repair_history_btn = gr.Button("修复历史记录", variant="secondary")
                    export_history_btn = gr.Button("导出历史记录", variant="secondary")
                    history_limit = gr.Slider(minimum=5, maximum=100, value=20, step=5, 
                                            label="显示记录数量", interactive=True)
                
                with gr.Column(scale=4):
                    history_table = gr.Dataframe(
                        headers=["时间", "问题", "答案", "相似度", "综合风险", "评估结果"],
                        datatype=["str", "str", "str", "number", "number", "str"],
                        col_count=(6, "fixed"),
                        row_count=10,
                        interactive=False,
                        wrap=True
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    similarity_chart = gr.Image(label="相似度分布", height=550)
                with gr.Column(scale=2):
                    risk_score_chart = gr.Image(label="综合风险分布", height=550)
        
        # 知识库管理标签页
        with gr.TabItem("知识库管理"):
            with gr.Row():
                with gr.Column():
                    kb_key_input = gr.Textbox(label="关键词", placeholder="例如：癌症")
                    kb_value_input = gr.Textbox(
                        label="知识内容", 
                        placeholder="例如：目前没有简单的治愈方法", 
                        lines=3
                    )
                    
                    with gr.Row():
                        kb_add_btn = gr.Button("添加/更新", variant="primary")
                        kb_delete_btn = gr.Button("删除", variant="stop")
                        kb_clear_btn = gr.Button("清除输入")
                
                with gr.Column():
                    kb_table = gr.Dataframe(
                        headers=["关键词", "知识内容"],
                        label="知识库内容"
                    )
                    kb_refresh_btn = gr.Button("刷新知识库")
            
            with gr.Row():
                kb_import_file = gr.File(label="导入知识库文件 (CSV)")
                kb_import_btn = gr.Button("导入")
                kb_export_btn = gr.Button("导出知识库")
        
        # 系统设置标签页
        with gr.TabItem("系统设置"):
            with gr.Row():
                with gr.Column():
                    similarity_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.get("SIMILARITY_THRESHOLD")),
                        step=0.05,
                        label="相似度阈值",
                        info="原始相似度低于此值会被标记为可能存在风险",
                        interactive=True
                    )
                    
                    # 添加风险评分阈值滑块
                    risk_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.get("RISK_THRESHOLD", 0.5)),
                        step=0.05,
                        label="风险评分阈值",
                        info="综合风险评分高于此值会被判定为高风险",
                        interactive=True
                    )
                    
                    # 添加相似度临界值滑块
                    similarity_critical = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.get("SIMILARITY_CRITICAL", 0.4)),
                        step=0.05,
                        label="相似度临界值",
                        info="相似度低于此值将直接判定为高风险，无视其他因素",
                        interactive=True
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=int(config.get("MAX_TOKENS")),
                        step=50,
                        label="最大生成Token数"
                    )
                    
                    model_name = gr.Dropdown(
                        choices=["hunyuan-turbo", "hunyuan-pro", "other-model"],
                        value=config.get("MODEL_NAME"),
                        label="模型名称"
                    )
                    
                    enable_enhancement = gr.Checkbox(
                        value=config.get("ENABLE_ENHANCEMENT", True),
                        label="启用混元模型增强功能"
                    )
                    
                    save_settings_btn = gr.Button("保存设置", variant="primary")
                
                with gr.Column():
                    system_stats = gr.JSON(label="系统统计")
                    refresh_stats_btn = gr.Button("刷新统计")
                    
                    generate_report_btn = gr.Button("生成系统报告")
                    system_report = gr.Textbox(label="系统报告", lines=10)
    
    # 定义更新函数
    def update_ui(question):
        """响应用户输入，更新界面"""
        try:
            # 验证答案
            result = verify_answer(question)
            
            # 提取结果
            answer = result.get("answer", "")
            reversed_question = result.get("reversed_question", "")
            similarity = result.get("similarity", 0.0)
            factuality = result.get("factuality_score", 5)
            is_risk = result.get("is_risk", True)
            risk_score = result.get("risk_score", 0.0)
            
            # 获取配置的阈值
            threshold = float(config.get("SIMILARITY_THRESHOLD", 0.65))
            risk_threshold = float(config.get("RISK_THRESHOLD", 0.5))
            sim_critical = float(config.get("SIMILARITY_CRITICAL", 0.4))
            
            # 设置风险指示器
            risk_value = {"高风险": 1.0, "低风险": 0.0} if is_risk else {"低风险": 1.0, "高风险": 0.0}
            
            # 修改阈值显示内容，使用最新获取的配置
            thresholds_md = f"风险评分阈值: {risk_threshold} (>{risk_threshold}为高风险) | 相似度临界值: {sim_critical} (<{sim_critical}直接判为高风险)"
            
            # 准备关键词表格
            keywords_data = [[kw] for kw in result.get("keywords", [])]
            
            # 准备反向问题表格
            reversed_questions_data = [[q] for q in result.get("all_reversed_questions", [])]
            
            # 在结果返回中添加检测方法图表
            detection_method_chart = None
            if "detection_methods" in result:
                detection_method_chart = create_detection_chart(result)
            
            # 格式化处理时间为纯秒数，不显示任何除法
            processing_time_str = f"{result.get('processing_time', 0):.2f}秒"
            
            # 返回更新的UI元素
            return [
                answer,
                reversed_question,
                similarity,
                factuality,
                risk_value,
                risk_score,  # 添加风险评分
                thresholds_md,  # 更新阈值说明
                result.get("knowledge_conflict_info", "无冲突"),
                keywords_data,
                reversed_questions_data,
                result.get("factuality_reason", ""),
                processing_time_str,  # 使用格式化后的处理时间字符串
                detection_method_chart
            ]
        except Exception as e:
            logger.error(f"更新界面失败: {str(e)}")
            return [
                f"系统错误: {str(e)}",
                "",
                0,
                0,
                {"低风险": 0.5, "高风险": 0.5},
                0.0,  # 风险评分
                f"风险评分阈值: {float(config.get('RISK_THRESHOLD', 0.5))} | 相似度临界值: {float(config.get('SIMILARITY_CRITICAL', 0.4))}",
                "",
                [],
                [],
                "",
                "0.00秒",  # 处理时间格式
                None
            ]
    
    def clear_inputs():
        """清除输入和输出"""
        # 获取风险阈值和相似度临界值来构建阈值说明
        risk_threshold = float(config.get('RISK_THRESHOLD', 0.5))
        sim_critical = float(config.get('SIMILARITY_CRITICAL', 0.4))
        thresholds_md = f"风险评分阈值: {risk_threshold} (>{risk_threshold}为高风险) | 相似度临界值: {sim_critical} (<{sim_critical}直接判为高风险)"
        
        # 返回列表而不是字典，确保返回14个值
        return [
            "",  # question_input (textbox)
            "",  # answer_output (textbox)
            "",  # reversed_output (textbox)
            0,   # similarity_score (number)
            0,   # factuality_score (number)
            {"低风险": 0.5, "高风险": 0.5},  # risk_indicator (label)
            0.0,  # risk_score_display (number)
            thresholds_md,  # thresholds_display (markdown)
            "",   # conflict_output (textbox)
            [],   # keywords_output (dataframe)
            [],   # all_reversed_output (dataframe)
            "",   # factuality_reason (textbox)
            "0.00秒",  # processing_time (textbox)
            None  # detection_method_chart (image)
        ]
    
    def load_history(limit_value):
        """加载历史记录"""
        try:
            logger.info(f"尝试加载历史记录, 限制数量: {limit_value}")
            history_file = os.path.join(config.get("DATA_DIR"), "history.csv")
            
            if not os.path.exists(history_file):
                logger.warning(f"历史记录文件不存在: {history_file}")
                return [], None, None
            
            # 创建表格数据和记录列表
            table_data = []
            records = []
            
            try:
                # 使用pandas读取CSV文件，处理复杂内容
                df = pd.read_csv(history_file, quoting=1, escapechar='\\', encoding='utf-8')
                logger.info(f"成功读取历史文件，记录数：{len(df)}")
                
                # 限制记录数量
                if limit_value > 0 and len(df) > limit_value:
                    df = df.head(limit_value)
                
                # 转换为记录列表
                records = df.to_dict('records')
                
                # 处理表格数据
                for item in records:
                    # 格式化时间
                    timestamp = item.get("timestamp", "")
                    time_str = timestamp.split("T")[0] + " " + timestamp.split("T")[1][:8] if "T" in timestamp else timestamp
                    
                    # 构建表格行
                    table_row = [
                        time_str,
                        item.get("question", "")[:50] + "..." if len(item.get("question", "")) > 50 else item.get("question", ""),
                        item.get("answer", "")[:100] + "..." if len(item.get("answer", "")) > 100 else item.get("answer", ""),
                        round(float(item.get("similarity", 0)), 4),  # 相似度
                        round(float(item.get("risk_score", 0)), 4),  # 添加风险评分
                        "高风险" if item.get("is_risk", False) else "低风险"
                    ]
                    table_data.append(table_row)
                
                logger.info(f"成功加载 {len(table_data)} 条历史记录")
                
                # 生成图表
                similarity_img = create_similarity_histogram(records)
                risk_img = create_risk_pie_chart(records)
                
                # 添加风险评分图表
                risk_score_chart_path = create_risk_score_histogram(records)
                
                return table_data, similarity_img, risk_score_chart_path
                
            except Exception as e:
                logger.error(f"读取历史记录失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 尝试备用方法
                try:
                    # 直接使用csv模块读取
                    with open(history_file, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        for i, row in enumerate(reader):
                            if limit_value > 0 and i >= limit_value:
                                break
                            
                            records.append(row)
                            
                            # 提取必要信息
                            timestamp = row.get('timestamp', '')[:16].replace('T', ' ')
                            question = str(row.get('question', '')).strip()
                            answer = str(row.get('answer', '')).strip()
                            if len(answer) > 100:
                                answer = answer[:100] + "..."
                            
                            similarity = 0.0
                            try:
                                similarity = float(row.get('similarity', 0))
                            except:
                                pass
                            
                            is_risk = str(row.get('is_risk', '')).lower() == 'true'
                            risk_text = "高风险" if is_risk else "低风险"
                            
                            table_data.append([
                                timestamp,
                                question,
                                answer,
                                similarity,
                                risk_text
                            ])
                    
                    # 生成图表
                    similarity_img = create_similarity_histogram(records)
                    risk_img = create_risk_pie_chart(records)
                    
                    # 添加风险评分图表
                    risk_score_chart_path = create_risk_score_histogram(records)
                    
                    return table_data, similarity_img, risk_score_chart_path
                except Exception as e2:
                    logger.error(f"备用读取方法也失败: {str(e2)}")
                    return [], None, None
        
        except Exception as e:
            logger.error(f"加载历史记录主函数失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [], None, None

    def create_empty_chart(title):
        """创建空图表"""
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "暂无数据", ha='center', va='center', fontsize=14)
        plt.title(title)
        
        # 保存图表
        img_path = os.path.join(config.get("DATA_DIR"), f"empty_{title}.png")
        plt.savefig(img_path, dpi=100)
        plt.close()
        return img_path

    def clear_history_data():
        """清除历史记录"""
        success = data_manager.clear_history(backup=True)
        if success:
            gr.Info("历史记录已清除")
        else:
            gr.Warning("清除历史记录失败")
        # 直接返回列表而非调用load_history
        return [], None, None
    
    def load_system_stats():
        """加载系统统计数据"""
        try:
            # 直接从数据管理器获取统计数据
            stats = data_manager.get_stats()
            
            # 详细记录返回值类型和内容
            logger.info(f"统计数据类型: {type(stats)}, 内容: {stats}")
            
            # 添加调试信息
            if stats is None:
                logger.warning("stats为None")
                return {"warning": "没有统计数据"}
            
            if isinstance(stats, str):
                logger.warning(f"stats是字符串: {stats}")
                return {"data": stats}
            
            # 确保返回有效的字典
            if not isinstance(stats, dict):
                return {
                    "警告": "统计数据格式不正确",
                    "类型": str(type(stats)),
                    "当前时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            return stats
        except Exception as e:
            logger.error(f"加载统计数据失败: {str(e)}")
            return {"错误": f"加载统计数据失败: {str(e)}"}
    
    def generate_system_report():
        """生成系统报告"""
        try:
            report = data_manager.generate_report()
            # 确保返回字符串而非None
            if report is None:
                return "无法生成系统报告"
            return report  # 这里应该返回字符串，用于显示在Textbox中
        except Exception as e:
            logger.error(f"生成系统报告失败: {str(e)}")
            return f"生成系统报告失败: {str(e)}"
    
    def save_system_settings(similarity_threshold, max_tokens, model_name, enable_enhancement, risk_threshold, similarity_critical):
        """保存系统设置"""
        try:
            # 更新配置
            config.set("SIMILARITY_THRESHOLD", similarity_threshold)
            config.set("MAX_TOKENS", max_tokens)
            config.set("MODEL_NAME", model_name)
            config.set("ENABLE_ENHANCEMENT", enable_enhancement)
            
            # 保存新增的风险评分相关设置
            config.set("RISK_THRESHOLD", risk_threshold)
            config.set("SIMILARITY_CRITICAL", similarity_critical)
            
            # 保存到配置文件
            config_file = os.path.join(config.get("DATA_DIR"), "config.json")
            success = config.save(config_file)
            
            if success:
                logger.info("系统设置已保存")
                return "系统设置已成功保存"
            else:
                logger.error("保存系统设置失败")
                return "保存系统设置失败，请查看日志"
        except Exception as e:
            logger.error(f"保存系统设置时出错: {str(e)}")
            return f"保存设置时出错: {str(e)}"
    
    # 刷新历史记录按钮的事件处理
    def refresh_history_data():
        """刷新历史记录"""
        try:
            logger.info("手动刷新历史记录")
            limit = int(history_limit.value) if history_limit and history_limit.value else 20
            result = load_history(limit)
            
            if result and result[0]:
                count = len(result[0])
                logger.info(f"刷新成功，加载了 {count} 条记录")
                gr.Info(f"已刷新历史记录，共 {count} 条")
            else:
                logger.warning("刷新历史记录后无数据")
                gr.Warning("无历史记录数据")
            
            return result
        except Exception as e:
            logger.error(f"刷新历史记录失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            gr.Warning(f"刷新失败: {str(e)}")
            return [], None, None

    # 绑定事件
    submit_btn.click(
        fn=update_ui,
        inputs=question_input,
        outputs=[
            answer_output,
            reversed_output,
            similarity_score,
            factuality_score,
            risk_indicator,
            risk_score_display,
            thresholds_display,
            conflict_output,
            keywords_output,
            all_reversed_output,
            factuality_reason,
            processing_time,
            detection_method_chart
        ]
    )
    
    clear_btn.click(
        fn=clear_inputs,
        outputs=[
            question_input,
            answer_output,
            reversed_output,
            similarity_score,
            factuality_score,
            risk_indicator,
            risk_score_display,
            thresholds_display,
            conflict_output,
            keywords_output,
            all_reversed_output,
            factuality_reason,
            processing_time,
            detection_method_chart
        ]
    )

    # 绑定历史记录相关事件
    refresh_history_btn.click(
        fn=refresh_history_data,
        outputs=[history_table, similarity_chart, risk_score_chart]
    )

    # 修改history_limit滑块的事件
    history_limit.change(
        fn=lambda x: load_history(x),
        inputs=[history_limit],
        outputs=[history_table, similarity_chart, risk_score_chart]
    )

    clear_history_btn.click(
        fn=clear_history_data,
        outputs=[history_table, similarity_chart, risk_score_chart]
    )

    export_history_btn.click(
        fn=lambda: f"历史记录已导出到 {os.path.join(config.get('DATA_DIR'), 'history_export.csv')}",
        outputs=gr.Textbox(visible=True)
    )

    # 绑定知识库相关事件
    kb_refresh_btn.click(
        fn=init_kb_table,
        outputs=[kb_table]
    )

    kb_add_btn.click(
        fn=add_knowledge_item,
        inputs=[kb_key_input, kb_value_input],
        outputs=[kb_key_input, kb_table]
    )

    kb_delete_btn.click(
        fn=delete_knowledge_item,
        inputs=[kb_key_input],
        outputs=[kb_key_input, kb_table]
    )

    kb_clear_btn.click(
        fn=clear_kb_inputs,
        outputs=[kb_key_input, kb_value_input]
    )

    kb_import_btn.click(
        fn=import_knowledge,
        inputs=[kb_import_file],
        outputs=[kb_table]
    )

    kb_export_btn.click(
        fn=export_knowledge,
        outputs=gr.Textbox(visible=True)
    )

    # 绑定系统设置相关事件
    refresh_stats_btn.click(
        fn=lambda: load_system_stats() or {"message": "没有统计数据"},  # 使用or操作符确保始终返回一个字典
        outputs=system_stats
    )

    generate_report_btn.click(
        fn=generate_system_report,
        outputs=[system_report]
    )

    save_settings_btn.click(
        fn=save_system_settings,
        inputs=[
            similarity_threshold, 
            max_tokens, 
            model_name, 
            enable_enhancement,
            risk_threshold,         # 添加风险阈值
            similarity_critical     # 添加相似度临界值
        ],
        outputs=gr.Textbox(visible=True)
    )

    # 修改应用启动时的初始化函数
    def on_app_start():
        """应用程序启动时的初始化"""
        logger.info("应用程序启动，初始化界面")
        # 加载历史记录
        history_result = load_history(20)  # 默认加载20条
        
        # 初始风险阈值和相似度临界值
        risk_threshold = float(config.get('RISK_THRESHOLD', 0.5))
        sim_critical = float(config.get('SIMILARITY_CRITICAL', 0.4))
        
        # 返回历史记录数据和空的表单
        return ["", "", 0, 0, {"低风险": 0.5, "高风险": 0.5}, 
                0.0,  # 风险评分初始值
                f"风险评分阈值: {risk_threshold} (>{risk_threshold}为高风险) | 相似度临界值: {sim_critical} (<{sim_critical}直接判为高风险)", 
                "", [], [], "", "0.00秒", None] + list(history_result)

    # 修改demo.launch之前，设置启动函数
    demo.load(
        fn=on_app_start,
        outputs=[
            question_input, answer_output, similarity_score, factuality_score, 
            risk_indicator, risk_score_display, thresholds_display, conflict_output, 
            keywords_output, all_reversed_output, factuality_reason, 
            processing_time, detection_method_chart,
            history_table, similarity_chart, risk_score_chart
        ]
    )

    # 添加历史记录修复按钮
    def clean_history_file():
        """清理历史记录文件，修复格式问题"""
        try:
            history_file = os.path.join(config.get("DATA_DIR"), "history.csv")
            
            if not os.path.exists(history_file):
                logger.warning(f"历史记录文件不存在: {history_file}")
                return False
            
            # 创建备份
            backup_file = f"{history_file}.bak"
            shutil.copy2(history_file, backup_file)
            
            # 读取数据
            with open(history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换可能导致解析问题的字符
            content = content.replace('""', '"')
            
            # 写回文件
            with open(history_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("已清理历史记录文件")
            return True
        except Exception as e:
            logger.error(f"清理历史记录文件失败: {str(e)}")
            return False

    # 在demo.load函数之前添加此调用
    clean_history_file()

    # 添加修复功能
    def repair_history():
        """修复历史记录文件"""
        success = clean_history_file()
        if success:
            gr.Info("历史记录已修复")
            return refresh_history_data()
        else:
            gr.Warning("修复历史记录失败")
            return [], None, None

    # 绑定修复按钮事件
    repair_history_btn.click(
        fn=repair_history,
        outputs=[history_table, similarity_chart, risk_score_chart]
    )

# 创建可视化函数
def create_similarity_histogram(history_data):
    """创建相似度分布直方图"""
    try:
        if not history_data:
            logger.warning("创建相似度直方图时历史数据为空")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "暂无数据", ha='center', va='center', fontsize=14)
            img_path = os.path.join(config.get("DATA_DIR"), "similarity_histogram.png")
            plt.savefig(img_path, dpi=100)
            plt.close()
            return img_path
        
        # 提取相似度数据，确保是有效的数字
        similarities = []
        for item in history_data:
            try:
                if 'similarity' in item and item['similarity'] is not None:
                    # 尝试转换为浮点数
                    sim = float(item['similarity'])
                    similarities.append(sim)
            except (ValueError, TypeError):
                continue
        
        if not similarities:
            logger.warning("没有有效的相似度数据")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "无有效相似度数据", ha='center', va='center', fontsize=14)
            img_path = os.path.join(config.get("DATA_DIR"), "similarity_histogram.png")
            plt.savefig(img_path, dpi=100)
            plt.close()
            return img_path
        
        logger.info(f"绘制相似度直方图，数据: {similarities[:5]}...")
        
        # 创建图表
        plt.figure(figsize=(12, 8))  # 增大图表尺寸
        if chinese_font:
            plt.hist(similarities, bins=10, alpha=0.7, color='#3498db', edgecolor='black')
            plt.axvline(x=sum(similarities) / len(similarities), color='red', linestyle='--', 
                        label=f'平均值: {sum(similarities) / len(similarities):.2f}')
            plt.xlabel('相似度值', fontproperties=chinese_font)
            plt.ylabel('频次', fontproperties=chinese_font)
            plt.title('相似度分布直方图', fontproperties=chinese_font)
            plt.grid(True, alpha=0.3)
            plt.legend(prop=chinese_font)
        else:
            # 使用英文
            plt.hist(similarities, bins=10, alpha=0.7, color='#3498db', edgecolor='black')
            plt.axvline(x=sum(similarities) / len(similarities), color='red', linestyle='--', 
                        label=f'Average: {sum(similarities) / len(similarities):.2f}')
            plt.xlabel('Similarity Value')
            plt.ylabel('Frequency')
            plt.title('Similarity Distribution')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 保存图表
        img_path = os.path.join(config.get("DATA_DIR"), "similarity_histogram.png")
        plt.savefig(img_path, dpi=150, bbox_inches='tight')  # 提高DPI并确保完整保存
        plt.close()
        
        return img_path
    except Exception as e:
        logger.error(f"创建相似度直方图失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 创建错误图表
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "图表生成失败", ha='center', va='center', fontsize=14, color='red')
        img_path = os.path.join(config.get("DATA_DIR"), "similarity_histogram_error.png")
        plt.savefig(img_path, dpi=100)
        plt.close()
        return img_path

def create_risk_pie_chart(history_data):
    """创建风险分布饼图"""
    if not history_data:
        # 添加调试信息
        logger.warning("创建风险饼图时历史数据为空")
        # 创建空图表
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "暂无数据", ha='center', va='center', fontsize=14)
        
        # 保存图表
        img_path = os.path.join(config.get("DATA_DIR"), "risk_pie_chart.png")
        plt.savefig(img_path, dpi=100)
        plt.close()
        return img_path
    
    # 计算风险统计
    safe_count = 0
    risk_count = 0
    
    for item in history_data:
        try:
            is_risk = item.get('is_risk')
            # 确保布尔值正确解析
            if isinstance(is_risk, str):
                is_risk = is_risk.lower() == 'true'
            
            if is_risk:
                risk_count += 1
            else:
                safe_count += 1
        except Exception as e:
            logger.error(f"解析风险数据出错: {str(e)}")
    
    # 检查是否有数据
    if safe_count == 0 and risk_count == 0:
        logger.warning("没有有效的风险数据")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "暂无有效数据", ha='center', va='center', fontsize=14)
        
        img_path = os.path.join(config.get("DATA_DIR"), "risk_pie_chart.png")
        plt.savefig(img_path, dpi=100)
        plt.close()
        return img_path
    
    # 创建饼图
    plt.figure(figsize=(12, 9))  # 增大饼图尺寸
    
    # 设置中文字体支持
    if chinese_font:
        labels = ['低风险', '高风险'] 
        plt.pie([safe_count, risk_count], 
                labels=labels, 
                colors=['#7cc576', '#f05454'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.03, 0.05),
                shadow=False,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                textprops={'fontproperties': chinese_font})
        
        plt.title('风险分布', fontsize=14, fontweight='bold', fontproperties=chinese_font)
        plt.suptitle(f'总计: {safe_count + risk_count}条记录', fontsize=10, y=0.05, 
                    fontproperties=chinese_font)
        
        # 添加风险评分说明
        plt.figtext(0.5, 0.01, 
                  "高风险: 风险评分>0.5或相似度<0.4，低风险: 其他情况", 
                  ha="center", fontproperties=chinese_font, fontsize=10)
    else:
        # 使用英文标签
        labels = ['Low Risk', 'High Risk']
        plt.pie([safe_count, risk_count], 
                labels=labels, 
                colors=['#7cc576', '#f05454'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.03, 0.05),
                shadow=False,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        
        plt.title('Risk Distribution', fontsize=14, fontweight='bold')
        plt.suptitle(f'Total: {safe_count + risk_count} records', fontsize=10, y=0.05)
        
        # 添加风险评分说明
        plt.figtext(0.5, 0.01, 
                  "High risk: score>0.5 or similarity<0.4, Low risk: otherwise", 
                  ha="center", fontsize=10)
    
    plt.axis('equal')
    
    # 保存图表
    img_path = os.path.join(config.get("DATA_DIR"), "risk_pie_chart.png")
    try:
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        logger.info(f"成功保存风险饼图: {img_path}")
    except Exception as e:
        logger.error(f"保存风险饼图失败: {str(e)}")
    finally:
        plt.close()
    
    return img_path

# 添加检测方法图表生成函数
def create_detection_chart(result):
    """创建幻觉检测方法对比图"""
    if not result or "detection_methods" not in result:
        return None
    
    methods = result["detection_methods"]
    
    # 准备数据
    method_names = []
    risk_scores = []
    weights = []
    
    for method, data in methods.items():
        method_names.append(method)
        risk_scores.append(data["score"])
        weights.append(data["weight"])
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 设置中文显示
    if chinese_font:
        # 方法名称中文映射
        method_mapping = {
            "reverse_question": "反向问题",
            "topic_drift": "主题漂移",
            "factuality": "事实性",
            "knowledge_conflict": "知识冲突"
        }
        
        # 使用中文方法名
        method_labels = [method_mapping.get(m, m) for m in method_names]
        
        # 绘制带权重的条形图
        bars = plt.bar(method_labels, risk_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # 添加权重标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.03, 
                    f'权重: {weights[i]:.2f}',
                    ha='center',
                    fontproperties=chinese_font)
        
        plt.xlabel('检测方法', fontproperties=chinese_font)
        plt.ylabel('风险分数', fontproperties=chinese_font)
        plt.title('幻觉检测方法风险评分', fontproperties=chinese_font)
        
        # 添加总体风险分数
        plt.axhline(y=result["risk_score"], color='r', linestyle='--')
        plt.text(len(method_names)/2, result["risk_score"] + 0.05, 
                f'总体风险: {result["risk_score"]:.2f}', ha='center',
                fontproperties=chinese_font)
    else:
        # 英文版本
        bars = plt.bar(method_names, risk_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.03, 
                    f'W: {weights[i]:.2f}',
                    ha='center')
        
        plt.xlabel('Detection Methods')
        plt.ylabel('Risk Score')
        plt.title('Hallucination Detection Risk Scores')
        
        plt.axhline(y=result["risk_score"], color='r', linestyle='--')
        plt.text(len(method_names)/2, result["risk_score"] + 0.05, 
                f'Overall Risk: {result["risk_score"]:.2f}', ha='center')
    
    plt.ylim(0, 1.1)  # 确保这行与上面的代码保持相同的缩进级别
    
    # 保存图表
    img_path = os.path.join(config.get("DATA_DIR"), "detection_methods.png")
    plt.savefig(img_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return img_path

# 添加风险评分分布图表函数
def create_risk_score_histogram(history_data):
    """创建风险评分分布直方图"""
    try:
        if not history_data:
            logger.warning("创建风险评分直方图时历史数据为空")
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, "暂无数据", ha='center', va='center', fontsize=14)
            img_path = os.path.join(config.get("DATA_DIR"), "risk_score_histogram.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()
            return img_path
        
        # 提取风险评分数据
        risk_scores = []
        for item in history_data:
            try:
                if 'risk_score' in item and item['risk_score'] is not None:
                    score = float(item['risk_score'])
                    risk_scores.append(score)
            except (ValueError, TypeError):
                continue
        
        if not risk_scores:
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, "无有效风险评分数据", ha='center', va='center', fontsize=14)
            img_path = os.path.join(config.get("DATA_DIR"), "risk_score_histogram.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()
            return img_path
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 获取风险阈值
        risk_threshold = float(config.get("RISK_THRESHOLD", 0.5))
        
        if chinese_font:
            plt.hist(risk_scores, bins=10, alpha=0.7, color='#f05454', edgecolor='black')
            plt.axvline(x=risk_threshold, color='red', linestyle='--', 
                      label=f'风险阈值: {risk_threshold:.2f}')
            plt.axvline(x=sum(risk_scores) / len(risk_scores), color='blue', linestyle='-.',
                      label=f'平均风险: {sum(risk_scores) / len(risk_scores):.2f}')
            plt.xlabel('风险评分', fontproperties=chinese_font, fontsize=12)
            plt.ylabel('频次', fontproperties=chinese_font, fontsize=12)
            plt.title('综合风险评分分布', fontproperties=chinese_font, fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(prop=chinese_font)
        else:
            plt.hist(risk_scores, bins=10, alpha=0.7, color='#f05454', edgecolor='black')
            plt.axvline(x=risk_threshold, color='red', linestyle='--', 
                      label=f'Risk Threshold: {risk_threshold:.2f}')
            plt.axvline(x=sum(risk_scores) / len(risk_scores), color='blue', linestyle='-.', 
                      label=f'Average: {sum(risk_scores) / len(risk_scores):.2f}')
            plt.xlabel('Risk Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Risk Score Distribution', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 添加解释文本
        if chinese_font:
            plt.figtext(0.5, 0.01, 
                      "风险评分说明: 由相似度(60%)、主题漂移(15%)、事实性(15%)和知识冲突(10%)加权计算", 
                      ha="center", fontproperties=chinese_font, fontsize=10)
        else:
            plt.figtext(0.5, 0.01, 
                      "Risk score: weighted by similarity(60%), topic drift(15%), factuality(15%), knowledge conflict(10%)", 
                      ha="center", fontsize=10)
        
        # 保存图表
        img_path = os.path.join(config.get("DATA_DIR"), "risk_score_histogram.png")
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return img_path
        
    except Exception as e:
        logger.error(f"创建风险评分直方图失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 创建错误图表
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "图表生成失败", ha='center', va='center', fontsize=14, color='red')
        img_path = os.path.join(config.get("DATA_DIR"), "risk_score_error.png")
        plt.savefig(img_path, dpi=100)
        plt.close()
        
        return img_path

if __name__ == "__main__":
    demo.launch(server_port=7860) 