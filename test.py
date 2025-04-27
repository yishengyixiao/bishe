import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()

from utils.api_interaction import generate_text
from utils.similarity_calculation import calculate_similarity

# 测试 API 调用
print("开始测试 API 调用...")
response = generate_text("你好，请简单介绍一下自己")
print(f"API 响应: {response}")

# 测试相似度计算
print("开始测试相似度计算...")
sim = calculate_similarity("你好吗", "你好")
print(f"相似度: {sim}")
