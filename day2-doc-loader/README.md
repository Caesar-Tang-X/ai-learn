# 一. 今日目标
昨天 Day1 我们搭好了推理网关，让程序能"和大模型对话"。但 RAG 的第一步其实不是对话，而是把知识喂给系统。这是整个 RAG 流水线的数据入口，做得好不好，直接决定后面检索、问答的质量。今天产出的模块，后面 Day3（Embedding）、Day4（向量库）会直接复用。

1. 今天要解决的是：
+ 把各种格式的文档（PDF / Word / Markdown）读进来 
+ 清洗掉噪声
+ 切成一块块适合检索的"知识片段（chunk）"
2. 今日三个核心能力：
+ 多格式文档加载：统一读取 PDF、Word、Markdown，输出纯文本
+ 文本清洗：去掉多余空行、乱码、页眉页脚、特殊符号等噪声
+ 分层分块（chunking）：按语义/长度把长文本切成带重叠的片段，为向量化做准备

# 二、先想清楚几个问题
在写代码前，先建立认知，这样写的时候才知道"为什么这么做"：

#### Q1：为什么不能把一整篇文档直接丢给大模型？
+ 大模型有上下文长度限制（token 上限）
+ 检索时需要"精准命中相关段落"，整篇文档粒度太粗，检索不准
+ 所以必须切成小块（chunk），一块块地做向量化和检索

#### Q2：为什么切块要"重叠（overlap）"？
+ 如果硬切，一句话可能被从中间截断，语义就断了
+ 让相邻两块有一部分重叠内容，避免关键信息被切碎丢失

#### Q3：什么是"分层分块"？
+ 不是简单按固定字数暴力切，而是优先按文档的自然结构切（先按段落 \n\n，再按句子 。！？，最后才按字符数硬切）
+ 这样切出来的每块尽量是一个完整语义单元

# 三、准备工作
## 步骤 1：创建项目目录
#### 1.1 创建项目文件夹
在你方便存放代码的磁盘新建文件夹，命名 `day2-doc-loader` 打开终端，cd 进入这个文件夹，示例（Windows cmd）：

```powershell
cd F:\ai-learn\day2-doc-loader
```

## 步骤 2：创建虚拟环境
#### 2.1 创建 Python 虚拟环境
执行命令：

```powershell
python -m venv venv
```

#### 2.2 激活虚拟环境
Windows CMD

```plain
venv\Scripts\activate
```

Windows PowerShell

```plain
.\venv\Scripts\activate
```

Mac / Linux

```plain
source venv/bin/activate
```

激活成功后，终端前缀会出现 `(venv)` 标识

#### 2.3 批量安装依赖包
激活环境后执行安装指令：

```plain
pip install pypdf python-docx markdown beautifulsoup4 langchain-text-splitters
```

等待全部依赖下载安装完成，无红色报错。

| 包 | 作用 |
| --- | --- |
| pypdf | 读取 PDF 文本 |
| python-docx | 读取 Word（.docx）文本 |
| markdown + beautifulsoup4 | Markdown 转 HTML 再提纯文本 |
| langchain-text-splitters | 提供成熟的递归分层切分器 |


# 四、开发实操
## 步骤 1：设计目录结构
```plain
day2-doc-loader/
├── main.py                 # 入口：演示整个"加载→清洗→分块"流水线
├── loaders/                # 文档加载层：负责读各种格式 → 纯文本
│   └── __init__.py
├── cleaners/               # 文本清洗层：去噪声
│   └── __init__.py
├── splitters/              # 分块层：把文本切成 chunk
│   └── __init__.py
└── samples/                # 测试素材（存在 pdf/docx/md）
```

## 步骤 2：编写 loaders/document_loader.py
#### 2.1 统一文档加载器，根据文件后缀自动选择对应解析方式，最终都输出纯文本
```python
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
```

#### 2.2 练习要点
+ 为什么 PDF 用 or "" 兜底？因为有的页 extract_text() 返回 None
+ Markdown 为啥要先转 HTML 再取文本，而不是直接读 .md 原文？因为直接读会带一堆 #、* 标记噪声，转 HTML 后 get_text() 自动剥离标签，更干净。

#### 2.3 遗留问题
+ DOCX 表格读不到

doc.paragraphs 只取了正文段落，Word 里的表格内容会被漏掉。Day2 阶段可以接受；以后要完整还原文档，需要遍历 doc.tables。先记着这个坑。

+ 文件不存在没兜底

如果传一个不存在的路径，PdfReader 会直接抛 FileNotFoundError，被上层捕获后不是友好提示。等你后面接全局异常时可以统一处理（Day1 的 BusinessException 思想可以复用）。

## 步骤 3：编写 cleaners/text_cleaner.py
#### 3.1 文本清洗器，去除多余空行、制表符、不可见字符等噪声
```python
import re

"""
author: tang
description: 文本清洗器，去除多余空行、制表符、不可见字符等噪声
"""

class TextCleaner:
    def clean(self, text: str) -> str:
        """清洗主流程：依次做多步正则处理"""
        text = self._remove_extra_whitespace(text)
        text = self._normalize_blank_lines(text)
        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        """把制表符、不间断空格、多余空格统一处理"""
        # 1. 把制表符 \t 换成普通空格
        text = text.replace("\t", " ")
        # 2. 把不间断空格 \xa0（PDF/HTML 常见）换成普通空格
        text = text.replace("\xa0", " ")
        # 3. 把任意连续多个空格（含全角空格）压缩成 1 个普通空格
        text = re.sub(r"[ \u3000]+", " ", text)
        return text

    def _normalize_blank_lines(self, text: str) -> str:
        """把连续 3 个以上的空行压缩成 1 个空行"""
        # 把 3 个及以上连续换行 (\n) 压成 2 个（即 1 个空行）
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 再去掉每行首尾的空白（清掉"空行里残留的空格"）
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text
```

#### 3.2 练习要点
+ \xa0（不换行空格）+ \u3000（全角空格）归一 真实 PDF/HTML 里最常见的"看不见的噪声"，用 .replace 先转成普通空格，再用 re.sub(r"[ \u3000]+", " ", text) 把连续空格压成 1 个。
+ 正则字符类 [ ] 同时匹配半角/全角空格 比单独匹配半角空格更稳，覆盖中文文档场景。
+ \n{3,} 压缩多空行 → 保留单空行 清掉噪声的同时不破坏段落边界，这是下游"分层分块"依赖的语义分隔符。
+ line.strip() 清掉空行里残留的空格 避免"看起来是空行、实际含空格"导致后续分块误判。
+ clean() 主流程按顺序串两步走 职责单一、顺序固定（先去空白噪声 → 再规范换行），符合分层解耦思想。

#### 2.3 遗留问题
+ 只做空白归一，没去页眉/页脚/页码/装饰线

扫描版 PDF 仍有结构噪声。可 按规则正则剔除页码行（如 ^第 \d+ 页$）。

+ 未做全角标点→半角、繁简转换等归一

同一词不同写法会分散检索。可 作为清洗增强步骤追加。

+ 清洗后未保留来源/段落索引元信息

出问题难溯源、难去重。可 在向量库写入时补 source/chunk_id 字段。

+ 文本含 \r\n（Windows 换行）时未统一

跨平台文件可能多出一个 \r。可 在 _remove_extra_whitespace 开头加 text = text.replace("\r\n", "\n").replace("\r", "\n")。

## 步骤 4：编写 splitters/recursive_splitter.py
#### 4.1 手写递归字符分层分块器，理解 chunking 原理
```python
import re

"""
author: tang
description: 手写递归字符分层分块器，理解 chunking 原理
"""

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # overlap 必须小于 chunk_size，否则下一箱永远带着上一箱全部内容，死循环
        assert chunk_overlap < chunk_size, "chunk_overlap 必须小于 chunk_size"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 分隔符按优先级从粗到细排列：先按段落，再按句子，最后按字符
        self.separators = ["\n\n", "\n", "。", "！", "？", "", ]

    def split(self, text: str) -> list[str]:
        """入口：先递归切成不超长的小片段，再合并成带重叠的 chunk"""
        pieces = self._split_recursive(text, self.separators)
        return self._merge_pieces(pieces)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """递归切分：优先用粗分隔符，超长段才降到更细粒度继续切"""
        # 取当前层分隔符，剩余的分隔符留给下一层递归（从粗到细逐级降级）
        sep = separators[0]
        rest = separators[1:]
        # 到最底层：按字符逐个切（保证任何超长内容最终都能被拆开）
        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)
        result = []
        for part in parts:
            # 当前片段够短 -> 直接收下
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # 仍超长 -> 用更细的分隔符继续递归切（rest 已去掉当前层）
                result.extend(self._split_recursive(part, rest))
        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """把小片段按顺序装箱，装满 chunk_size 就封箱，并在下一箱开头保留 overlap 个字符"""
        chunks = []
        current = ""
        for piece in pieces:
            # 加上这个 piece 会超出 -> 先把 current 封箱
            if len(current) + len(piece) > self.chunk_size and current:
                chunks.append(current)
                # 新箱开头 = 上一箱尾部 overlap 个字符（实现"重叠"）
                current = current[-self.chunk_overlap:] + piece
            else:
                current += piece
        # 循环结束，把最后一箱也收进去
        if current:
            chunks.append(current)
        return chunks
```

#### 4.2 练习要点
+ separators 从粗到细的优先级设计 ["\n\n" → "\n" → "。" → "！" → "？" → ""]，先按段落、再按句子、最后按字符——这就是"分层"的本质：尽量保留自然语义边界。
+ 递归时传 rest = separators[1:] 逐级降级 超长片段才用更细分隔符继续切，绝不回头用已消耗掉的粗分隔符，避免无限递归。
+ 空字符串 "" 兜底 当所有标点分隔符都切不开时，退化到按字符切，保证任何超长内容终能被拆到 ≤ chunk_size，递归必然终止。
+ __init__ 里的 assert overlap < chunk_size 安全阀 防止 overlap ≥ chunk_size 导致下一箱永远搬不走上一箱内容、死循环。
+ _merge_pieces 的"装箱 + 重叠" current = current[-chunk_overlap:] + piece 把上一箱尾部 overlap 字符复制到下一箱开头，实现语义衔接。
+ 循环结束补最后一箱 if current: chunks.append(current)，只有"溢出才封箱"，漏掉这行会丢数据，是经典边界坑。

#### 4.3 遗留问题
+ 分隔符被丢弃（split(sep) 吃掉 。等）

合并时片段可能粘连，如"苹果好吃"。可 用 keep_separator 思路保留标点（LangChain 版对照可见）。

+ 长度用 len() 按字符计，未支持 token

中文/英文混合时块大小与模型上下文不完全对应。可 length_function 接 tokenizer。

+ 未过滤空片段

可能出现空 chunk 污染检索。可 合并前 if piece.strip(): result.append(...)。

+ 重叠按"尾部字符"切，可能切断一个词/标点

overlap 衔接处偶尔不自然。可 按词边界或保留完整句尾再做 overlap。

+ 无 source / chunk_id 元信息

出问题难溯源、向量库无法回链原文。可 写入向量库时附 metadata。

+ 纯同步、单文件

大文件/多文件批量处理慢。可 改为批处理或流式。

## 步骤 5：编写 main.py
#### 5.1 文档处理流水线入口：加载 → 清洗 → 分块
```python
from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from splitters.recursive_splitter import RecursiveTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
author: tang
description: 文档处理流水线入口：加载 → 清洗 → 分块
"""

def mine_run_pipeline(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    手写递归字符分层分块器
    """
    print(f"\n===== 手写递归字符分层分块器 =====")
    # 1. 加载
    loader = DocumentLoader()
    raw_text = loader.load(file_path)
    print(f"[加载] 原始字符数: {len(raw_text)}")
    # 2. 清洗
    cleaner = TextCleaner()
    clean_text = cleaner.clean(raw_text)
    print(f"[清洗] 清洗后字符数: {len(clean_text)}")
    # 3. 分块
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split(clean_text)
    print(f"[分块] 共切出 {len(chunks)} 块 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    # 4. 打印每块预览
    for i, c in enumerate(chunks):
        print(f"\n===== Chunk {i+1} (长度 {len(c)}) =====")
        print(c[:200])  # 只预览前 200 字，避免刷屏
    return chunks

def lc_run_pipeline(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    LangChain分层分块器
    """
    print(f"\n===== LangChain分层分块器 =====")
    # 1. 加载
    loader = DocumentLoader()
    raw_text = loader.load(file_path)
    print(f"[加载] 原始字符数: {len(raw_text)}")
    # 2. 清洗
    cleaner = TextCleaner()
    clean_text = cleaner.clean(raw_text)
    print(f"[清洗] 清洗后字符数: {len(clean_text)}")
    # 3. 分块
    lc = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ""],
        keep_separator=True,
    )
    chunks = lc.split_text(clean_text)
    # 4. 打印每块预览
    for i, c in enumerate(chunks):
        print(f"\n===== Chunk {i+1} (长度 {len(c)}) =====")
        print(c[:200])  # 只预览前 200 字，避免刷屏
    return chunks


if __name__ == "__main__":
    mine_run_pipeline("samples/README.md", chunk_size=300, chunk_overlap=50)
    lc_run_pipeline("samples/README.md", chunk_size=300, chunk_overlap=50)
```

# 五. 总结：
从零实现了**工程化文档处理流水线**，解决 RAG 数据入口"读不了多格式、脏文本干扰检索、长文档无法送进模型"的痛点，产出结构化知识片段可直接复用于后续向量化。

#### 1. 技术栈
Python + pypdf + python-docx + markdown + beautifulsoup4 + langchain-text-splitters

#### 2. 三大核心模块功能
1. **统一文档加载 loaders/document_loader.py**
    - 按后缀自动路由 PDF / Word / Markdown 三种解析器
    - 统一输出纯文本，上层无需关心来源格式
    - PDF `None` 页兜底、Markdown 先转 HTML 再取文本，规避常见读取坑
2. **文本清洗 cleaners/text_cleaner.py**
    - 归一 `\t` / `\xa0` / 全角空格，消除"看不见的噪声"
    - 压缩连续空行但保留单空行，不破坏段落边界
    - 清洗后字符数从 `12889` 降到 `12159`，去除约 730 字噪声
3. **分层分块 splitters/recursive_splitter.py**
    - 从粗到细递归切分：优先段落 → 句子 → 字符兜底
    - 合并时相邻块保留 overlap，避免语义被切断
    - 手写版与 LangChain 版块数基本一致（54 vs 55），证明原理正确

#### 3. 分层工程化思想
```plain
loaders(加载) → cleaners(清洗) → splitters(分块) → main(编排)
```

+ 解耦：换文档格式只改 loaders；改切分策略只动 splitters，互不影响
+ 可扩展：新增格式（如 .txt/.html）只需加一个 `_load_xxx` 方法
+ 可验证：手写版理解原理，LangChain 版对照验证，工程与原理兼得

# 六、关键知识点理解复盘
#### Q1：为什么不能直接把整篇文档丢给大模型做 RAG？
大模型有上下文长度上限；且检索时需要"精准命中相关段落"，整篇粒度太粗检索不准。必须切成小块（chunk）分别向量化、检索。

#### Q2：为什么分块要"重叠（overlap）"？
硬切可能把一个完整句子从中间切断，语义断裂。让相邻两块共享尾部 overlap 个字符，保证这句话无论落在哪块都能被检索命中。

#### Q3：什么是"分层分块"？和暴力按字数切有什么区别？
暴力切按固定字符数硬切，常切断句子。分层切优先用自然结构分隔符（段落 `\n\n` → 句子 `。！？` → 字符兜底），尽量让每块是完整语义单元。

#### Q4：递归切分时为什么每次都传 `separators[1:]`，而不是每次从头用全部分隔符？
超长片段需降到更细层级继续切，但那段里已没有更粗的分隔符（如已无 `\n\n`），再 split 等于没切，会无限递归。每次消耗当前层、只把更细的往下传，保证逐级降级、必然终止。

#### Q5：为什么 `separators` 末尾要有空字符串 `""`？
当所有标点分隔符都切不开（如超长无标点文本）时，空串代表"按字符切"，保证任何内容终能被拆到 ≤ chunk_size，递归不会栈溢出。

#### Q6：为什么 `chunk_overlap` 必须小于 `chunk_size`？
若 overlap ≥ chunk_size，下一箱开头永远带着上一箱几乎全部内容，内容清不掉，chunk 数量爆炸甚至死循环。`__init__` 里的 `assert` 就是这道安全阀。

#### Q7：手写版和 LangChain 版的实质差距在哪？
核心思路一致，差距在细节打磨：LangChain 用 `keep_separator=True` 保留分隔符（避免"苹果好吃"粘连）、支持 `length_function` 按 token 计长、自动过滤空 chunk。手写版胜在"懂原理"，LangChain 版胜在"工程完备"。

# 七、遗留问题与待补强技术债
| # | 问题 | 影响 | 计划解决时机 |
| --- | --- | --- | --- |
| 1 | DOCX 表格读不到（`doc.paragraphs` 只取正文） | Word 表格内容丢失 | 补 `doc.tables` 遍历；或 Day15 改造 Dify 解析器 |
| 2 | PDF 页眉/页脚/页码未剔除 | 扫描版 PDF 结构噪声残留 | 进阶正则剔除页码行（如 `^第 \d+ 页$`） |
| 3 | 未剥离 Markdown 代码围栏（```python / plain / powershell） | 分块混入源码与标记词，干扰 Embedding 检索 | 给 `TextCleaner` 加代码围栏清洗步骤（推荐加餐练习） |
| 4 | 手写版分隔符被丢弃，片段可能粘连 | chunk 可读性略差 | 采用 LangChain 版或保留分隔符 |
| 5 | 长度按字符计，未支持 token | 中英混合时块大小与模型上下文不完全对应 | 接 tokenizer，Day3 后补课 |
| 6 | 未过滤空片段、无 source/chunk_id 元信息 | 检索难溯源、难去重、向量库无法回链原文 | Day4 写入向量库时附 `metadata` |
| 7 | 文件不存在/格式损坏抛原生异常，无友好提示 | 调用方拿不到统一报错 | 接 Day1 的 `BusinessException` 统一异常 |