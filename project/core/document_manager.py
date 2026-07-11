# 文档管理：接收上传的 PDF/MD，转换、切块、写入向量库和父块存储。

from pathlib import Path
import shutil
import config
from utils import pdfs_to_markdowns, clear_directory_contents


class DocumentManager:
    """🅱️ 文档的增删查，是界面操作和底层存储之间的中间层。"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)   # 目录不存在就建

    def add_documents(self, document_paths, progress_callback=None):
        """🅱️ 接收文件路径列表，逐个处理，返回 (新增数, 跳过数)。

        处理流程（每个文件）：查重 → PDF转MD/复制MD → 切块 → 存父块 → 存子块向量 → 出错回滚。
        ⚠️ 落盘顺序：先存父块，再存子块向量。这样向量入库失败时能靠父块 id 清理，不留垃圾。
        输入：document_paths（路径或路径列表）、progress_callback（可选进度回调）
        输出：(added, skipped)
        """
        # if：没传任何文件 → 直接返回 0,0
        if not document_paths:
            return 0, 0

        # 如果传进来的是单个字符串，包成列表统一处理
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        # 过滤：只留下非空、且后缀是 .pdf / .md 的路径（列表推导里的 if 做筛选）
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]

        # if：过滤完一个合法文件都不剩 → 返回 0,0
        if not document_paths:
            return 0, 0

        added = 0
        skipped = 0

        for i, doc_path in enumerate(document_paths):
            # if：界面传了进度回调 → 汇报当前进度（第几个/共几个）
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")

            source_path = Path(doc_path)
            doc_name = source_path.stem
            md_path = self.markdown_dir / f"{doc_name}.md"

            # if：同名 md 已存在 → 跳过（不覆盖，防止重复索引），计数 +1 后进入下个文件
            if md_path.exists():
                skipped += 1
                continue

            parent_ids = []
            try:
                # 步骤1：拿到 Markdown —— if 是 .md 直接复制，else 是 PDF 就转换
                if source_path.suffix.lower() == ".md":
                    shutil.copy(source_path, md_path)
                else:
                    pdfs_to_markdowns(str(source_path), overwrite=False)

                # 步骤2：切块（得到父块和子块）
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(
                    md_path,
                    source_name=source_path.name,
                )
                # if：一个子块都没切出来 → 视为失败，抛错进入下面的回滚
                if not child_chunks:
                    raise ValueError("No child chunks were created.")

                # 步骤3：先存父块（JSON 文件），并记下所有 parent_id 以便出错回滚
                parent_ids = [parent_id for parent_id, _ in parent_chunks]
                self.rag_system.parent_store.save_many(parent_chunks)

                # 步骤4：再存子块向量（会调用智谱嵌入 + 写入 Qdrant）
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                collection.add_documents(child_chunks)

                added += 1

            except Exception as e:
                # 任一步失败 → 回滚：删掉已写的父块文件
                self.rag_system.parent_store.delete_many(parent_ids)
                # if：md 文件已经生成了 → 也一并删掉，保持干净
                if md_path.exists():
                    md_path.unlink()
                print(f"处理文档出错 {doc_path}: {e}")
                skipped += 1

        return added, skipped

    def get_markdown_files(self):
        """🅲 获取知识库中所有文档的来源文件名列表（界面显示用）。"""
        sources = self.rag_system.parent_store.list_sources()
        # if：父块存储里查到了来源名 → 直接用
        if sources:
            return sources
        # 兜底：父块存储为空时，直接扫 markdown_dir 目录里的 .md 文件名
        return sorted(p.name for p in self.markdown_dir.glob("*.md"))

    def clear_all(self):
        """🅲 清空知识库：删向量集合 + 清 markdown_dir + 清 parent_store，再建空集合。"""
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
        clear_directory_contents(self.markdown_dir)
        self.rag_system.parent_store.clear_store()
        # 重建一个空集合，方便用户立刻上传新文档（不用重启）
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)
