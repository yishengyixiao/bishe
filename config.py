import os
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础配置
BASE_CONFIG = {
    "API_KEY": os.getenv("HUNYUAN_API_KEY"),
    "API_BASE": "https://api.hunyuan.cloud.tencent.com/v1",
    "MODEL_NAME": os.getenv("MODEL_NAME", "hunyuan-turbo"),
    "MAX_TOKENS": int(os.getenv("MAX_TOKENS", 100)),
    "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 0.65)),
    "RETRY_ATTEMPTS": 3,
    "LOG_LEVEL": "INFO",
    "DATA_DIR": "data",
    "LOG_DIR": "logs",
    "ENABLE_ENHANCEMENT": True,  # 混元模型的增强功能
    "RISK_THRESHOLD": float(os.getenv("RISK_THRESHOLD", 0.5)),
    "SIMILARITY_CRITICAL": float(os.getenv("SIMILARITY_CRITICAL", 0.4)),
}

# 确保必要的目录存在
for directory in [BASE_CONFIG["DATA_DIR"], BASE_CONFIG["LOG_DIR"]]:
    os.makedirs(directory, exist_ok=True)

# 高级配置
class Config:
    def __init__(self, config_file=None):
        self.config = BASE_CONFIG.copy()
        
        # 如果提供了配置文件，从文件加载配置
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
            except Exception as e:
                print(f"加载配置文件失败: {str(e)}")
    
    def get(self, key, default=None):
        """获取配置项"""
        value = self.config.get(key, default)
        # 添加对关键配置项的打印
        if key in ["API_KEY", "MODEL_NAME", "SIMILARITY_THRESHOLD", "MAX_TOKENS"]:
            masked_value = value
            if key == "API_KEY" and value:
                # 掩码API密钥，只显示前4位和后4位
                masked_value = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
            print(f"配置: {key} = {masked_value}")
        return value
    
    def set(self, key, value):
        """设置配置项"""
        old_value = self.config.get(key)
        self.config[key] = value
        
        # 记录配置更改
        if old_value != value:
            print(f"配置已更改: {key} = {value} (原值: {old_value})")
        
        return value
    
    def save(self, config_file):
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {str(e)}")
            return False

# 创建全局配置实例
config = Config()
