# 文档切块器：把 Markdown 文档切成「父块(大块,给上下文)」和「子块(小块,给检索)」。
#
# 为什么要切两层？
#   子块(500字)：小 → 向量相似度搜索更精准
#   父块(2000~4000字)：大 → 模型读到的上下文更完整
#   子块里存 parent_id → 搜到子块后，再去拿对应的父块，兼顾精准和完整

import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class DocumentChunker:
    """文档切块器：按标题切父块 → 整理大小 → 从父块切子块 → 给子块打 parent_id 标记。"""

    def __init__(self):
        # 启动时就校验配置是否合法，避免运行到一半才报错（fail fast）
        # if①：父块尺寸配置不合理（最小值<=0，或最大值比最小值还小）→ 直接报错
        if config.MIN_PARENT_SIZE <= 0 or config.MAX_PARENT_SIZE < config.MIN_PARENT_SIZE:
            raise ValueError("父块大小配置错误：MIN_PARENT_SIZE 必须 > 0 且 <= MAX_PARENT_SIZE")
        # if②：子块重叠必须落在 [0, 子块大小) 区间内（重叠比块还大就没意义）
        if not 0 <= config.CHILD_CHUNK_OVERLAP < config.CHILD_CHUNK_SIZE:
            raise ValueError("子块重叠配置错误：CHILD_CHUNK_OVERLAP 必须 < CHILD_CHUNK_SIZE")
        # if③：子块重叠也不能大于父块最大值（否则拆分逻辑会乱）
        if config.CHILD_CHUNK_OVERLAP >= config.MAX_PARENT_SIZE:
            raise ValueError("子块重叠不能大于父块最大值")

        # 父块切分器：按 Markdown 标题(# ## ###)切分，保留标题文本（strip_headers=False）
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON,
            strip_headers=False
        )
        # 子块切分器：按字符数切，保留重叠（让相邻子块有共同内容，搜索不会漏掉边界句子）
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE   # 父块最小字数
        self.__max_parent_size = config.MAX_PARENT_SIZE   # 父块最大字数

    @staticmethod
    def __merge_metadata(target, source, prepend=False):
        """🅲 合并两个块的 metadata（标题路径）。遇到相同的 key 就拼成"H1 -> H2"。"""
        for key, value in source.items():
            # if：target 里还没有这个键 → 直接搬过去
            if key not in target:
                target[key] = value
            # else：两边都有这个键 → 把两段标题路径拼起来并去重
            else:
                # prepend 决定谁在前：True 表示 source 的值放前面（用于"并入后一个块"的场景）
                first, second = (value, target[key]) if prepend else (target[key], value)
                values = [
                    item.strip()
                    for raw in (first, second)
                    for item in str(raw).split(" -> ")
                    if item.strip()                       # 过滤掉空字符串
                ]
                # dict.fromkeys 去重保序（同 append_unique 的语法，见 graph_state.py）
                target[key] = " -> ".join(dict.fromkeys(values))

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        """🅲 遍历目录下所有 .md 文件，全部切块，返回(所有父块列表, 所有子块列表)。"""
        all_parent_chunks, all_child_chunks = [], []
        # 遍历目录下每个 .md 文件，逐个切块后汇总
        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)       # extend = 把列表元素逐个追加
            all_child_chunks.extend(child_chunks)
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path, source_name=None):
        """🅱️ 对单个 Markdown 文件完整切块，返回(父块列表, 子块列表)。

        流程：读文件 → 按标题切父块 → 合并过小 → 拆分过大 → 清理残余 → 切子块并打 parent_id
        输入：md_path（.md 文件路径），source_name（来源文件名，默认用 pdf 原名）
        输出：(all_parent_chunks, all_child_chunks)
        """
        doc_path = Path(md_path)
        # source_name 没传就用文件名 + .pdf 后缀（a or b：a 为空才用 b）
        source_name = source_name or f"{doc_path.stem}.pdf"

        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())   # 先按标题切成碎块

        merged_parents  = self.__merge_small_parents(parent_chunks)       # 合并太小的
        split_parents   = self.__split_large_parents(merged_parents)      # 拆分太大的
        cleaned_parents = self.__clean_small_chunks(split_parents)        # 清理残余小块

        # if：整理完后仍有超大块 → 说明配置或逻辑有问题，报错（any=只要有一个满足就True）
        if any(len(chunk.page_content) > self.__max_parent_size for chunk in cleaned_parents):
            raise ValueError("切块后存在超过 MAX_PARENT_SIZE 的父块，请检查配置。")

        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path, source_name)
        return all_parent_chunks, all_child_chunks

    def __merge_small_parents(self, chunks):
        """🅱️ 把连续的小块（< MIN_PARENT_SIZE）贪心合并，直到够大为止。

        输入：按标题切出来的原始块列表（可能很碎）
        输出：合并后的块列表（每块尽量 >= MIN_PARENT_SIZE）
        """
        # if：空列表直接返回，避免后面 current 相关逻辑出错
        if not chunks:
            return []
        merged, current = [], None      # merged=结果列表，current=正在累积的块
        for chunk in chunks:
            # if：还没有正在累积的块 → 拿当前块开个头
            if current is None:
                current = chunk
            # else：已有累积块 → 把当前块的正文和 metadata 并进去
            else:
                current.page_content += "\n\n" + chunk.page_content
                self.__merge_metadata(current.metadata, chunk.metadata)
            # if：累积够大了 → 收进结果，清空 current 重新开始下一块
            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        # if：循环结束还剩一个没够大的 current（尾巴）
        if current:
            # if：结果里已有块 → 把尾巴并入最后一块（避免产生一个孤零零的小块）
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                self.__merge_metadata(merged[-1].metadata, current.metadata)
            # else：整篇文档就这么点内容 → 尾巴自成一块
            else:
                merged.append(current)
        return merged

    def __split_large_parents(self, chunks):
        """🅱️ 把超过 MAX_PARENT_SIZE 的块用字符切分器再拆小。

        输入：合并后的块列表（可能有超大块）
        输出：不超过 MAX_PARENT_SIZE 的块列表
        """
        split_chunks = []
        for chunk in chunks:
            # if：块没超过上限 → 原样保留
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            # else：块太大 → 临时建一个字符切分器把它再切小
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                split_chunks.extend(splitter.split_documents([chunk]))
        return split_chunks

    def __rebalance_pair(self, first, second):
        """🅱️ 当相邻两块一大一小时，找合适的断点把它们重新均衡地切开。

        输入：两个相邻块
        输出：重新切分后的两个块（都尽量接近 MIN_PARENT_SIZE）
        先合并，再在中间找最近的段落/行/词边界切开，避免切在句子中间。
        """
        combined = first.page_content.rstrip() + "\n\n" + second.page_content.lstrip()   # 先拼成一整段
        lower = max(1, len(combined) - self.__max_parent_size)   # 切点允许的最左位置
        upper = min(self.__max_parent_size, len(combined) - 1)   # 切点允许的最右位置
        # if：合并后足够长（能切成两个都达标的块）→ 把切点范围再收紧到都不低于最小值
        if len(combined) >= 2 * self.__min_parent_size:
            lower = max(lower, self.__min_parent_size)
            upper = min(upper, len(combined) - self.__min_parent_size)
        preferred = min(max(len(combined) // 2, lower), upper)   # 理想切点：正中间，但夹在[lower,upper]内

        split_at = preferred
        # 从理想切点附近，优先找"段落分隔"，再找"换行"，最后找"空格"，尽量不切断句子
        for separator in ("\n\n", "\n", " "):
            before = combined.rfind(separator, lower, preferred + 1)   # 理想点左侧最近的分隔符
            after  = combined.find(separator, preferred, upper + 1)    # 理想点右侧最近的分隔符
            # if：左侧找到了合法分隔符 → 用它，跳出
            if before >= lower:
                split_at = before; break
            # if：右侧找到了 → 用它，跳出
            if after != -1:
                split_at = after; break

        left_text  = combined[:split_at].rstrip()    # 切点左边 = 第一块
        right_text = combined[split_at:].lstrip()     # 切点右边 = 第二块
        # if：本该切成两个达标块，但按分隔符切完却有一块不达标 → 放弃"按分隔符"，改回正中间硬切
        if len(combined) >= 2 * self.__min_parent_size and (
            len(left_text) < self.__min_parent_size or len(right_text) < self.__min_parent_size
        ):
            split_at = preferred
            left_text, right_text = combined[:split_at], combined[split_at:]
        # if：切出来有一边是空的 → 这次再平衡失败，原样退回两块
        if not left_text or not right_text:
            return first, second

        metadata = dict(first.metadata)               # 两块合并后共用一份 metadata
        self.__merge_metadata(metadata, second.metadata)
        first.page_content,  first.metadata  = left_text,  dict(metadata)
        second.page_content, second.metadata = right_text, dict(metadata)
        return first, second

    def __clean_small_chunks(self, chunks):
        """🅱️ 清理仍然过小的块：优先并入相邻块，实在不行做再平衡。"""
        cleaned = []
        for i, chunk in enumerate(chunks):
            # if：这个块太小，需要处理
            if len(chunk.page_content) < self.__min_parent_size:
                # if①：前一个块（cleaned[-1]）还装得下 → 并入前一个（+2 是那个"\n\n"分隔的长度）
                if cleaned and len(cleaned[-1].page_content) + 2 + len(chunk.page_content) <= self.__max_parent_size:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self.__merge_metadata(cleaned[-1].metadata, chunk.metadata)
                # elif②：前一个装不下，但后一个块装得下 → 并入后一个（prepend=True 表示放到后一个前面）
                elif (i < len(chunks) - 1
                      and len(chunk.page_content) + 2 + len(chunks[i+1].page_content) <= self.__max_parent_size):
                    chunks[i+1].page_content = chunk.page_content + "\n\n" + chunks[i+1].page_content
                    self.__merge_metadata(chunks[i+1].metadata, chunk.metadata, prepend=True)
                # else③：前后都装不下 → 只能先保留这个小块（宁可有小块也不超大）
                else:
                    cleaned.append(chunk)
            # else：块够大 → 直接保留
            else:
                cleaned.append(chunk)

        # 第二轮：对仍然过小的块做一次"再平衡"（跟相邻块借内容匀一匀）
        for i, chunk in enumerate(cleaned):
            # if：这块已经够大，或整个列表就一块（没法再平衡）→ 跳过
            if len(chunk.page_content) >= self.__min_parent_size or len(cleaned) == 1:
                continue
            # if：不是最后一块 → 和"后一块"再平衡
            if i < len(cleaned) - 1:
                cleaned[i], cleaned[i+1] = self.__rebalance_pair(chunk, cleaned[i+1])
            # else：是最后一块 → 和"前一块"再平衡
            else:
                cleaned[i-1], cleaned[i] = self.__rebalance_pair(cleaned[i-1], chunk)
        return cleaned

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path, source_name):
        """🅱️ 给每个父块编 ID，再从父块切出子块，子块继承 parent_id。

        这里是"分层检索的命门"：子块 metadata 里有 parent_id → 搜到子块后知道去哪取父块。
        输入：空列表×2（收集结果用）、父块列表、文档路径、来源文件名
        输出：无（结果追加到传入的列表里）
        """
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_p{i}"          # 编号，例如 "公司手册_p0"
            # 把来源文件名和 parent_id 写进父块的 metadata
            p_chunk.metadata.update({"source": source_name, "parent_id": parent_id})

            all_parent_pairs.append((parent_id, p_chunk))  # (id, 文档块) 对，之后存到 parent_store
            all_child_chunks.extend(                        # 从这个父块切出子块（子块自带 parent_id）
                self.__child_splitter.split_documents([p_chunk])
            )
