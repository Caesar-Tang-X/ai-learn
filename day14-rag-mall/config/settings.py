"""
集中配置：系统唯一配置源。
配置项以「模块/服务」为前缀，便于区分来源；敏感信息（API Key）优先从环境变量/.env 读取。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 本地 Ollama 服务
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "bge-m3"          # 嵌入式模型
    ollama_embedding_dimension: int = 1024
    ollama_llm_model: str = "qwen2.5:3b"            # 生成模型

    # 阿里百炼服务
    alibaba_dashscope_api_key: str = ""
    alibaba_dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    alibaba_embedding_model: str = "qwen3.7-text-embedding"   # 嵌入式模型
    alibaba_embedding_dimension: int = 1024
    alibaba_rerank_model: str = "qwen3-rerank"                # 重排模型
    alibaba_llm_model: str = "qwen3.8-max"                    # 生成模型

    # 模型提供方
    embedding_provider: str = "alibaba"         # 嵌入式模型：ollama / alibaba
    rerank_provider: str = "alibaba"            # 重排模型：alibaba（留空表示不启用重排）
    llm_provider: str = "alibaba"               # 生成模型：ollama / alibaba

    # 检索参数
    retrieval_top_k: int = 20                   # 每路召回的候选数
    retrieval_final_top_k: int = 20             # 融合后送入重排的候选数
    rerank_top_n: int = 10                      # 重排后最终保留条数（默认输出数量）
    vector_threshold: float = 0.20              # 向量余弦相似度下限，低于视为无关
    rerank_threshold: float = 0.2               # 重排分数绝对下限（过低会放过跨类目噪声）
    rerank_relative_threshold: float = 0.6      # 重排相对下限：分数 < 最高分×该比例 视为不相关
    min_relevance_score: float = 0.45           # 最高分低于此值视为「无相关商品」，直接返回空

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "root"
    postgres_password: str = "123456"
    postgres_db: str = "ragdb"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = "123456"

    # 会话记忆
    memory_keep_recent: int = 8
    memory_max_tokens: int = 2500
    session_ttl_seconds: int = 86400

    # 相对约束推导：用户表达「太贵了/更便宜」时，价格上限 = 上一轮最高价 × 该系数（仅当用户未显式给预算）
    relative_price_factor: float = 0.7
    # 相对降价的价格下限保护：price_min = 上一轮最低价 × 该系数，避免召回极端低价（如几毛钱）垃圾品
    relative_price_floor: float = 0.5
    # 相对上提（如「太便宜了」）的价格下限抬高：price_min = 上一轮最低价 × 该系数，往更高价位走
    relative_price_lift: float = 2.0

    # 预算区间处理（单位：比例，作用于「元」预算）
    # 模糊/区间预算（如「1千左右」「300-600」）在 DB 硬过滤时的外扩比例：
    # 下限 × (1-expand)、上限 × (1+expand)，避免候选被卡死在过窄区间。
    budget_expand_ratio: float = 0.4
    # 候选不足时的【渐进放宽】系数序列（成对：下限倍率、上限倍率），逐档尝试；
    # 始终围绕原始预算保留范围，绝不放大到近乎无约束。例 (0.5,2.0) 即下限×0.5、上限×2.0。
    budget_relax_ratios: tuple = (0.5, 2.0, 0.3, 3.0)

    @property
    def postgres_database_url(self) -> str:
        """psycopg 连接串：postgresql://user:password@host:port/dbname"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Redis 连接串：redis://[:password]@host:port/db"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
