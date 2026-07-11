# 父块存储：把父块以 JSON 文件形式存到本地，靠文件名当主键读写。
# 每个父块存一个文件：parent_store/公司手册_p0.json

import re
import json
import config
from utils import clear_directory_contents
from pathlib import Path
from typing import List, Dict


class ParentStoreManager:
    """B 级 父块的本地文件存储。

    为什么不存 Qdrant？
    Qdrant 只存向量 + 简短 metadata，存完整大块文本浪费且慢。
    父块用普通 JSON 文件存，靠 parent_id 做文件名当主键，读写简单直接。
    """

    __store_path: Path

    def __init__(self, store_path=config.PARENT_STORE_PATH):
        self.__store_path = Path(store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        """B 级 把一个父块存成 JSON 文件。

        输入：parent_id（如 "公司手册_p0"），content（正文），metadata（来源等信息）
        输出：无（写文件）
        文件格式：{"page_content": "...", "metadata": {...}}
        ensure_ascii=False：保留中文，不转义成 uXXXX
        """
        file_path = self.__store_path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def save_many(self, parents: List) -> None:
        """C 级 批量保存父块，parents 是 [(parent_id, Document), ...] 列表。"""
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    def delete_many(self, parent_ids: List[str]) -> None:
        """C 级 批量删除父块文件（索引失败时回滚用）。"""
        for parent_id in parent_ids:
            file_path = self.__store_path / f"{parent_id}.json"
            # if：文件存在才删（不存在就跳过，避免报错）
            if file_path.exists():
                file_path.unlink()

    def load(self, parent_id: str) -> Dict:
        """C 级 读取并解析 JSON 文件，返回原始 dict。"""
        file_path = self.__store_path / (
            parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        )
        return json.loads(file_path.read_text(encoding="utf-8"))

    def load_content(self, parent_id: str) -> Dict:
        """B 级 按 parent_id 读取父块，返回标准格式 {content, parent_id, metadata}。

        输入：parent_id（字符串）
        输出：{"content": 正文, "parent_id": id, "metadata": {...}}
        工具 retrieve_parent_chunks 调用这里来取大块内容。
        """
        data = self.load(parent_id)
        return {
            "content": data["page_content"],
            "parent_id": parent_id,
            "metadata": data["metadata"]
        }

    @staticmethod
    def _get_sort_key(id_str):
        """C 级 从 parent_id 里提取序号用于排序。"公司手册_p3" -> 3。"""
        match = re.search(r'_(?:parent_|p)(\d+)$', id_str)
        return int(match.group(1)) if match else 0

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        """B 级 按一批 parent_id 读取父块，自动去重并按块序号排序。

        输入：parent_id 列表（可能有重复）
        输出：去重排序后的 [{content, parent_id, metadata}, ...] 列表
        """
        unique_ids = set(parent_ids)
        return [self.load_content(pid) for pid in sorted(unique_ids, key=self._get_sort_key)]

    def list_sources(self) -> List[str]:
        """B 级 列出知识库里所有来源文件名（界面"当前文档"列表用）。

        输出：去重排序的来源文件名列表，如 ["公司手册.pdf", "财务制度.pdf"]
        """
        sources = set()
        for file_path in self.__store_path.glob("*.json"):
            try:
                source = json.loads(file_path.read_text(encoding="utf-8")).get("metadata", {}).get("source")
                # if：这个文件有 source 字段才收集（去重靠 set）
                if source:
                    sources.add(source)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(sources)

    def clear_store(self) -> None:
        """C 级 清空父块存储目录（清空知识库时调用）。"""
        self.__store_path.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.__store_path)
