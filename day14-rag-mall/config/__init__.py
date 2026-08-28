"""
配置包门面：对外暴露 Settings 配置类与 get_settings 单例。
"""
from config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
