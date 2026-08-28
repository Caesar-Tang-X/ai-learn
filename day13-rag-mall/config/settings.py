"""
集中配置：系统唯一的配置来源（单一配置源）。

所有模块都通过 get_settings() 获取配置。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    从 .env 读取的应用配置，带类型校验
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    embedding_dimension: int = 1024
    llm_model: str = "qwen2.5:3b"

    # PostgreSQL (PGVector)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag"
    postgres_password: str = "rag123"
    postgres_db: str = "ragdb"

    # 检索参数
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 8
    rerank_top_n: int = 4

    @property
    def database_url(self) -> str:
        """
        拼出 psycopg3 使用的异步连接串
        格式：postgresql://user:password@host:port/dbname
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """
    返回全局唯一的 Settings 实例（带缓存，避免重复读取文件）
    """
    return Settings()
