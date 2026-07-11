# 通用工具函数：PDF 转 Markdown、清空目录、估算 token 数。

import os
import shutil
import config
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob
import tiktoken
from functools import lru_cache   # 缓存装饰器：函数结果只算一次，之后直接返回缓存

# fastembed 内部用 tokenizers 库，多进程时会警告。设为 false 关掉。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clear_directory_contents(directory: Path) -> None:
    """🅲 清空目录下所有内容，但保留目录本身。"""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)   # 删整个子目录
        else:
            child.unlink()         # 删单个文件


def pdf_to_markdown(pdf_path, output_dir):
    """🅱️ 把单个 PDF 转成 Markdown 文件，存到 output_dir。

    输入：pdf_path（PDF 文件路径），output_dir（输出目录）
    输出：在 output_dir 里生成同名 .md 文件（无返回值）

    注意 overwrite 不在这里控制，由上层 pdfs_to_markdowns 决定。
    """
    doc = pymupdf.open(pdf_path)
    # to_markdown 把 PDF 的字号、缩进、标题排版转成 # ## ### 这样的 Markdown 结构
    # ignore_images=True：不提取图片（图片内容不能被向量化，跳过）
    # page_separators=True：每页末尾插一行分隔，便于定位（但会产生脏数据，知道即可）
    md = pymupdf4llm.to_markdown(
        doc,
        header=False, footer=False,
        page_separators=True,
        ignore_images=True,
        write_images=False,
        image_path=None
    )
    # surrogatepass/ignore：处理 PDF 里偶尔出现的非法 UTF-8 字节，防止写文件时崩
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))


def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    """🅱️ 批量把 PDF 转成 Markdown，存到 config.MARKDOWN_DIR。

    输入：
        path_pattern : 文件路径或 glob 表达式，如 "docs/*.pdf" 或 "doc.pdf"
        overwrite    : False = 已有同名 .md 就跳过（默认，防止重复转换）
                       True  = 强制重新转

    ⚠️ 坑：overwrite=False 意味着你改了 PDF 但忘了删旧 md，系统会继续用旧的。
    换文档后记得先在界面点"清空全部"。
    """
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)   # 目录不存在就创建

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():        # 覆盖模式 或 文件不存在，才转
            pdf_to_markdown(pdf_path, output_dir)


@lru_cache(maxsize=1)
def _get_token_encoding():
    """🅲 获取 tiktoken 编码器，只初始化一次（lru_cache 缓存结果）。

    tiktoken 是 OpenAI 的 token 计数工具。
    这里用 gpt-4 的编码（cl100k_base），和中文模型的分词不完全一致，
    但用来估算量级够了。
    """
    try:
        return tiktoken.encoding_for_model("gpt-4")
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None   # 加载失败则返回 None，上层用字符数估算


def estimate_context_tokens(messages: list) -> int:
    """🅱️ 估算一批消息的 token 数，用于判断是否需要压缩上下文。

    输入：消息对象列表（HumanMessage / AIMessage 等）
    输出：估算的 token 总数（整数）

    这把"卡尺"不准：
      ① 跳过了 content="" 的消息（工具调用指令没有 content，被忽略）
      ② 没算 system prompt
      ③ tiktoken 是英文分词器，对中文低估
    但它只用来触发压缩，偏低一点没关系（宁可晚点压缩也比压崩强）。
    """
    # 只取有内容的消息的文本
    contents = [
        str(msg.content)
        for msg in messages
        if hasattr(msg, "content") and msg.content
    ]
    encoding = _get_token_encoding()
    if encoding is None:
        # tiktoken 加载失败时的降级方案：
        # 中文约 1~2 字符/token，用 //2 估算（英文经验值 //4 对中文低估太多）
        return sum(max(1, len(content) // 2) for content in contents)
    # tiktoken 精确计数（对英文准，对中文偏低）
    return sum(len(encoding.encode(content)) for content in contents)
