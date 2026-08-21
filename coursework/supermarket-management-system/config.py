import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')

if load_dotenv and os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


class Config:
    """基础配置"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    BASE_DIR = BASE_DIR
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'data', 'supermarket.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 日志配置
    LOG_FILE = os.getenv('LOG_FILE', os.path.join(BASE_DIR, 'log', 'app.log'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', str(5 * 1024 * 1024)))
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

    # 大模型配置
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'siliconflow')
    LLM_API_KEY = os.getenv('LLM_API_KEY', '')
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.siliconflow.cn/v1')
    LLM_CHAT_PATH = os.getenv('LLM_CHAT_PATH', '/chat/completions')
    # 推荐：硅基流动上的通用高质量模型
    LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-ai/DeepSeek-V3')
    LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '30'))
    # 可选：网站信息（用于平台侧识别来源）
    LLM_SITE_URL = os.getenv('LLM_SITE_URL', '')
    LLM_SITE_NAME = os.getenv('LLM_SITE_NAME', 'supermarket-management-system')

    # 开发模式下禁用模板缓存
    TEMPLATES_AUTO_RELOAD = True

