import os
import tempfile
import logging
import time
import json
from datetime import datetime
import re
import requests
from config import config

# 配置日志
logger = logging.getLogger(__name__)

# 添加终端处理器，确保日志同时输出到终端
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

def generate_text(prompt, temperature=0.7, retries=None, model=None, max_tokens=None):
    """
    通过混元模型API生成文本
    
    参数:
        prompt: 输入的提示文本
        temperature: 温度参数
        retries: 重试次数
        model: 模型名称
        max_tokens: 最大token数
        
    返回:
        生成的文本
    """
    try:
        # 获取配置，使用传入的参数或默认配置
        api_key = config.get("API_KEY")
        api_base = config.get("API_BASE")
        model_name = model or config.get("MODEL_NAME")
        max_tokens_value = max_tokens or int(config.get("MAX_TOKENS"))
        retry_count = retries or config.get("RETRY_ATTEMPTS", 3)
        
        if not api_key:
            logger.error("未设置API密钥，请在.env文件中配置HUNYUAN_API_KEY")
            return "未配置API密钥，请检查系统设置"
        
        # 记录API调用尝试
        logger.info(f"尝试通过混元API生成文本: model={model_name}, prompt长度={len(prompt)}, temperature={temperature}")
        # 添加终端输出提示内容
        print(f"\n=== API调用 ===\n提示内容: {prompt[:100]}..." if len(prompt) > 100 else f"\n=== API调用 ===\n提示内容: {prompt}")
        
        # 构建请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens_value
        }
        
        # 添加增强功能标志（如果已启用）
        if config.get("ENABLE_ENHANCEMENT", True):
            payload["enhancement"] = True
        
        print(f"发送请求: {api_base}/chat/completions")
        print(f"请求参数: model={model_name}, temperature={temperature}, max_tokens={max_tokens_value}")
        
        # 重试逻辑
        for attempt in range(retry_count):
            try:
                start_time = time.time()
                response = requests.post(f"{api_base}/chat/completions", json=payload, headers=headers)
                elapsed_time = time.time() - start_time
                
                print(f"API响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # 记录成功
                    logger.info(f"成功生成文本: 长度={len(text)}, 耗时={elapsed_time:.2f}秒")
                    
                    # 在终端显示生成的文本
                    print(f"\n=== 模型回答 ({elapsed_time:.2f}秒) ===\n{text}\n")
                    
                    # 保存API调用记录
                    save_api_call_record(prompt, text, model_name, elapsed_time)
                    
                    return text
                else:
                    logger.warning(f"API调用失败(尝试 {attempt+1}/{retry_count}): 状态码={response.status_code}, 响应={response.text}")
                    print(f"API调用失败: 状态码={response.status_code}")
                    print(f"错误响应: {response.text}")
                    
                    # 如果配额超限或服务器错误，等待后重试
                    if response.status_code in [429, 500, 502, 503, 504]:
                        wait_time = (attempt + 1) * 2
                        logger.info(f"等待 {wait_time} 秒后重试...")
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        # 其他错误直接退出重试
                        break
            
            except Exception as e:
                logger.error(f"请求错误(尝试 {attempt+1}/{retry_count}): {str(e)}")
                print(f"请求错误: {str(e)}")
                time.sleep(2)
        
        logger.error(f"已达到最大重试次数({retry_count})，生成文本失败")
        print(f"已达到最大重试次数({retry_count})，生成文本失败")
        return f"API调用失败，请稍后重试。错误信息: 已达到最大重试次数({retry_count})"
        
    except Exception as e:
        logger.error(f"生成文本过程中发生错误: {str(e)}")
        print(f"生成文本过程中发生错误: {str(e)}")
        return f"生成文本时出错: {str(e)}"

def generate_multiple_responses(prompt, n=3, temperature=0.7):
    """生成多个不同的回答"""
    responses = []
    for i in range(n):
        temp = temperature + (i * 0.1)
        response = generate_text(prompt, temperature=temp)
        if response:
            responses.append(response)
    return responses

def check_factuality(statement):
    """检查陈述的事实性"""
    prompt = f"""请评估以下陈述的可信度，给出1-10的评分（1表示完全不可信，10表示非常可信）和简短理由：

陈述：{statement}

回答格式：
评分：[数字]
理由：[简短解释]"""

    response = generate_text(prompt, temperature=0.3)
    
    if not response:
        return 5, "无法评估"
    
    try:
        # 尝试提取评分
        score_match = re.search(r'评分：?(\d+)', response)
        score = int(score_match.group(1)) if score_match else 5
        
        # 尝试提取理由
        reason_match = re.search(r'理由：?(.*)', response)
        reason = reason_match.group(1).strip() if reason_match else "无详细理由"
        
        return score, reason
    except:
        return 5, "解析评估结果失败"

def save_api_call_record(prompt, response, model, elapsed_time):
    """保存 API 调用记录到 JSON 文件"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "prompt": prompt,
        "response": response,
        "elapsed_time": elapsed_time
    }
    
    try:
        # 使用配置目录
        log_dir = config.get("LOG_DIR")
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用当前日期作为文件名
        filename = os.path.join(log_dir, f"api_calls_{datetime.now().strftime('%Y%m%d')}.json")
        
        # 读取现有记录
        records = []
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                records = json.load(f)
        
        # 添加新记录
        records.append(record)
        
        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"保存 API 调用记录失败: {str(e)}")

def analyze_sentiment(text):
    """
    分析文本情感
    
    参数:
        text: 输入文本
    
    返回:
        情感分析结果 (positive, neutral, negative)
    """
    prompt = f"请分析以下文本的情感倾向，只回答'积极'、'中性'或'消极'：\n\n{text}"
    response = generate_text(prompt, temperature=0.3, max_tokens=10)
    
    if not response:
        return "neutral"
    
    response = response.lower()
    if "积极" in response:
        return "positive"
    elif "消极" in response:
        return "negative"
    else:
        return "neutral" 