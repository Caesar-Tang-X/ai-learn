import os
from pypdf import PdfReader
from docx import Document
from markdown import markdown
from bs4 import BeautifulSoup

"""
author: tang
description: 统一文档加载器，支持 PDF / Word / Markdown → 纯文本
"""

class DocumentLoader:
    def load(self, file_path: str) -> str:
        """入口：根据后缀分发到对应解析方法，返回纯文本"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".docx":
            return self._load_docx(file_path)
        elif ext in (".md", ".markdown"):
            return self._load_markdown(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _load_pdf(self, file_path: str) -> str:
        """解析 PDF 文件"""
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text

    def _load_docx(self, file_path: str) -> str:
        """解析 Word 文件"""
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text

    def _load_markdown(self, file_path: str) -> str:
        """解析 Markdown 文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            html = markdown(f.read())
        text = BeautifulSoup(html, "html.parser").get_text()
        return text
