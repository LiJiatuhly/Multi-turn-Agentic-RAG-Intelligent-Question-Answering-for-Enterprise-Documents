"""【评测环境】用 RAGAS 从 markdown_docs/ 自动生成中文测试集。

流程：
    语料 → KnowledgeGraph（LLM 抽标题/摘要/实体/主题 + 向量建边）
         → 出题器适配成中文 → 生成 (问题, 标准答案, 来源原文) → testset.jsonl

⚠️ 这一步会打很多次模型，30 道题大概几百次调用，先用 --size 5 试跑。
⚠️ 知识图谱会缓存到 artifacts/knowledge_graph.json，第二次跑默认直接复用，
   改了语料才需要 --rebuild-kg。

这里的 KnowledgeGraph 不是你的线上 Qdrant 检索库，而是 Ragas 出题阶段的
辅助图谱：Ragas 先从文档抽取标题、摘要、实体和主题，再用 embedding 计算
节点之间的相似关系。出题器沿着这些关系生成单跳、多跳等问题，并同时生成
标准答案和参考原文。之后人工审核测试集，真正的 RAG 运行由 run_rag.py 负责。

用法：
    python gen_testset.py --size 5          # 先小样跑通
    python gen_testset.py --size 30         # 正式生成
    python gen_testset.py --rebuild-kg      # 语料变了，重建知识图谱
"""

import argparse
import asyncio
import csv
import json
import random
import sys
import warnings

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from openai import OpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import apply_transforms, default_transforms

import eval_config as C
from zhipu_embed import ZhipuEmbeddings
from eval_utils import (
    corpus_fingerprint,
    file_sha256,
    package_versions,
    stable_question_id,
    utc_now,
    validate_testset,
    write_json_atomic,
    write_jsonl_atomic,
)

# LangchainEmbeddingsWrapper 在 0.4 里标了 deprecated，但它是目前唯一
# 同时满足「出题 transforms 要 embed_text」和「ResponseRelevancy 要 embed_query」的类，
# 官方源码里也仍有专门的 legacy 分支处理它，所以照用，只是把警告静音。
warnings.filterwarnings("ignore", category=DeprecationWarning)

def build_models():
    """造出题用的 LLM 与嵌入模型（都走智谱 OpenAI 兼容接口）。

    LLM 负责抽取图谱信息和生成问题；embedding 负责把文档/节点变成向量，
    让知识图谱能够建立“哪些内容语义相近”的边。
    """
    client = OpenAI(api_key=C.API_KEY, base_url=C.BASE_URL, timeout=120.0, max_retries=3)
    llm = llm_factory(
        C.GENERATOR_MODEL_ID,
        client=client,
        max_tokens=C.GEN_MAX_TOKENS,
    )
    embeddings = LangchainEmbeddingsWrapper(
        ZhipuEmbeddings(
            model=C.EMBEDDING_MODEL,
            api_key=C.API_KEY,
            base_url=C.BASE_URL,
            batch_size=C.EMBEDDING_BATCH_SIZE,
        )
    )
    return llm, embeddings


def load_docs():
    """读 markdown_docs/ 下的 .md。

    ⚠️ 必须显式指定 TextLoader + utf-8：
       DirectoryLoader 默认用 UnstructuredFileLoader，既要额外装包，中文还容易乱码。
    """
    if not C.MARKDOWN_DIR.exists():
        raise SystemExit(f"找不到语料目录：{C.MARKDOWN_DIR}（先把文档放进去，或跑一次 PDF 转 Markdown）")
    loader = DirectoryLoader(
        str(C.MARKDOWN_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        raise SystemExit(f"{C.MARKDOWN_DIR} 里没有 .md 文件。")
    print(f"载入 {len(docs)} 篇文档，共 {sum(len(d.page_content) for d in docs)} 字")
    return docs


def build_or_load_kg(docs, llm, embeddings, rebuild: bool) -> KnowledgeGraph:
    """建知识图谱；已有缓存就直接读（这一步最费时费钱）。

    图谱的生命周期是：文档节点 -> transforms 抽取属性/建立关系 -> 保存缓存。
    manifest 同时记录语料哈希；语料变化时拒绝复用旧图，避免测试集不可复现。
    """
    current_corpus = corpus_fingerprint(C.MARKDOWN_DIR)
    if C.KG_PATH.exists() and not rebuild:
        if not C.KG_MANIFEST.exists():
            raise SystemExit("知识图谱缺少 manifest，无法确认语料版本；请加 --rebuild-kg 重建。")
        cached = json.loads(C.KG_MANIFEST.read_text(encoding="utf-8"))
        if cached.get("corpus", {}).get("sha256") != current_corpus["sha256"]:
            raise SystemExit("语料已变化，但知识图谱还是旧版本；请加 --rebuild-kg 重建。")
        kg = KnowledgeGraph.load(str(C.KG_PATH))
        print(f"复用已有知识图谱：{kg}")
        return kg

    # 先把每篇原始文档作为 DOCUMENT 节点放进图里；后续 transforms 会在此基础上
    # 增加摘要、实体、主题等节点或属性，并建立语义关系。
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )
    print(f"初始图谱：{kg}，开始跑 transforms（抽标题/摘要/实体/主题 + 算相似度，比较慢）……")
    # Ragas 内置 transforms 是知识图谱的“加工流水线”，会调用 LLM 和 embedding。
    apply_transforms(
        kg,
        default_transforms(documents=docs, llm=llm, embedding_model=embeddings),
        run_config=RunConfig(
            max_workers=C.JUDGE_MAX_WORKERS,
            timeout=C.JUDGE_TIMEOUT,
            max_retries=2,
            seed=C.RANDOM_SEED,
        ),
    )
    kg.save(str(C.KG_PATH))
    write_json_atomic(C.KG_MANIFEST, {
        "schema_version": 1,
        "created_at": utc_now(),
        "corpus": current_corpus,
        "model": C.GENERATOR_MODEL_ID,
        "embedding_model": C.EMBEDDING_MODEL,
        "random_seed": C.RANDOM_SEED,
        "knowledge_graph_sha256": file_sha256(C.KG_PATH),
        "versions": package_versions(("ragas", "langchain-core", "openai", "instructor")),
    })
    print(f"图谱已保存：{C.KG_PATH} → {kg}")
    return kg


def adapt_to_chinese(distribution, llm):
    """把出题 prompt 适配成中文。

    不做这一步，英文 prompt 出中文题，问法会很别扭、reference 也容易夹英文。
    """
    async def _run():
        for synthesizer, _weight in distribution:
            prompts = await synthesizer.adapt_prompts(C.TESTSET_LANGUAGE, llm=llm)
            synthesizer.set_prompts(**prompts)
    asyncio.run(_run())
    print(f"出题 prompt 已适配为：{C.TESTSET_LANGUAGE}")


def dump(testset):
    """落盘：jsonl 给机器读，csv 给人眼看。

    testset.jsonl 是 Ragas 生成的候选集；人工在 preview.csv 填“是/否”后，
    curate_testset.py 才会生成 run_rag.py 真正消费的 testset_curated.jsonl。
    """
    raw_rows = testset.to_list()
    rows = []
    for row in raw_rows:
        question = str(row.get("user_input", "")).strip()
        reference = str(row.get("reference", "")).strip()
        rows.append({
            "id": stable_question_id(question, reference),
            "user_input": question,
            "reference": reference,
            "reference_contexts": row.get("reference_contexts", []),
            "synthesizer": row.get("synthesizer_name", ""),
        })
    rows = validate_testset(rows)
    write_jsonl_atomic(C.TESTSET_PATH, rows)

    with open(C.TESTSET_PREVIEW, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "保留", "问题", "标准答案", "出题器", "审核备注", "来源原文（截断）"])
        for row in rows:
            writer.writerow([
                row["id"],
                "",
                row["user_input"],
                row["reference"],
                row.get("synthesizer", ""),
                "",
                " ||| ".join(c[:120] for c in row.get("reference_contexts", [])),
            ])
    write_json_atomic(C.TESTSET_MANIFEST, {
        "schema_version": 1,
        "created_at": utc_now(),
        "testset_sha256": file_sha256(C.TESTSET_PATH),
        "corpus": corpus_fingerprint(C.MARKDOWN_DIR),
        "requested_size": len(raw_rows),
        "valid_size": len(rows),
        "model": C.GENERATOR_MODEL_ID,
        "embedding_model": C.EMBEDDING_MODEL,
        "language": C.TESTSET_LANGUAGE,
        "random_seed": C.RANDOM_SEED,
        "versions": package_versions(("ragas", "langchain-core", "openai", "instructor")),
    })
    print(f"\n已生成 {len(rows)} 道题：\n  {C.TESTSET_PATH}\n  {C.TESTSET_PREVIEW}")
    print("👉 强烈建议先打开 csv 人工过一遍，把不通顺、答非所问的题删掉再往下跑。")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=C.TESTSET_SIZE, help="生成多少道题")
    parser.add_argument("--rebuild-kg", action="store_true", help="强制重建知识图谱")
    parser.add_argument("--no-adapt", action="store_true", help="跳过中文适配（省钱，但出题质量下降）")
    args = parser.parse_args()

    if not C.API_KEY:
        raise SystemExit("没读到 API_KEY，请检查 project/.env")

    random.seed(C.RANDOM_SEED)

    llm, embeddings = build_models()
    docs = load_docs()
    kg = build_or_load_kg(docs, llm, embeddings, rebuild=args.rebuild_kg)

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)
    distribution = default_query_distribution(llm, kg)
    print("出题器分布：" + ", ".join(f"{s.__class__.__name__}({w})" for s, w in distribution))

    if not args.no_adapt:
        adapt_to_chinese(distribution, llm)

    testset = generator.generate(
        testset_size=args.size,
        query_distribution=distribution,
        num_personas=C.NUM_PERSONAS,
        run_config=RunConfig(
            max_workers=C.JUDGE_MAX_WORKERS,
            timeout=C.JUDGE_TIMEOUT,
            max_retries=2,
            seed=C.RANDOM_SEED,
        ),
        raise_exceptions=False,
    )
    dump(testset)


if __name__ == "__main__":
    main()
