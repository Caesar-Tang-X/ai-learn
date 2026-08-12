"""
文档加载：从文件提取纯文本。

支持 .txt / .md（直接读）与 .pdf（用 pypdf）。
返回文档纯文本；不支持的类型抛 ValueError。
"""
import os


def load_document(path: str) -> str:
    """
    读取单个文件，返回其纯文本内容。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = os.path.splitext(path)[1].lower()
    
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".pdf":
        return _load_pdf(path)

    raise ValueError(f"不支持的文件类型: {ext}")


def _load_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
