# 评测端统一配置。
#
# ⚠️ 这个文件被【两个环境】共用（项目环境跑 run_rag.py，评测环境跑 gen_testset.py / score.py），
#    所以这里只用标准库：不 import ragas，也不 import 项目里的任何模块。

import os
from pathlib import Path

# ============================================================
# --- 目录 ---
# 约定：本目录 evaluation/ 与 project/ 平级，都在仓库根目录下
# ============================================================
EVAL_DIR = Path(__file__).resolve().parent          # evaluation/
REPO_ROOT = EVAL_DIR.parent                         # 仓库根目录
PROJECT_DIR = REPO_ROOT / "project"                 # 你的源码目录
MARKDOWN_DIR = REPO_ROOT / "markdown_docs"          # 语料（与 config.py 里的 MARKDOWN_DIR 一致）
ARTIFACT_DIR = EVAL_DIR / "artifacts"               # 所有中间产物都落在这里
ARTIFACT_DIR.mkdir(exist_ok=True)

KG_PATH = ARTIFACT_DIR / "knowledge_graph.json"     # RAGAS 知识图谱缓存（生成测试集最贵的一步）
KG_MANIFEST = ARTIFACT_DIR / "knowledge_graph_manifest.json"
TESTSET_PATH = ARTIFACT_DIR / "testset.jsonl"       # 自动生成的测试集
TESTSET_PREVIEW = ARTIFACT_DIR / "testset_preview.csv"   # 同一份测试集，给人眼看的
TESTSET_CURATED_PATH = ARTIFACT_DIR / "testset_curated.jsonl"  # 人工确认后，跑图只读这份
TESTSET_MANIFEST = ARTIFACT_DIR / "testset_manifest.json"
CURATED_MANIFEST = ARTIFACT_DIR / "curated_manifest.json"
RUNS_PATH = ARTIFACT_DIR / "runs.jsonl"             # 跑图采集到的原始结果（可断点续跑）
RUNS_PREVIEW = ARTIFACT_DIR / "runs_preview.md"     # 同一批真实回答的人类可读版
SCORES_TURN = ARTIFACT_DIR / "scores_turn.csv"      # 整轮级评分明细
SCORES_SUB = ARTIFACT_DIR / "scores_sub.csv"        # 子问题级评分明细
SCORES_AGENT = ARTIFACT_DIR / "scores_agent.csv"    # Agent 行为明细（不调用裁判模型）
REPORT_PATH = ARTIFACT_DIR / "report.md"            # 汇总报告
RUN_MANIFEST = ARTIFACT_DIR / "run_manifest.json"


# ============================================================
# --- .env 读取（不依赖 python-dotenv，评测环境少装一个包）---
# ============================================================
def _load_env(path: Path) -> None:
    """把 .env 里的 KEY=VALUE 塞进 os.environ（已存在的不覆盖）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(PROJECT_DIR / ".env")


# ============================================================
# --- 模型 ---
# ============================================================
API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
MODEL_ID = os.environ.get("MODEL_ID", "glm-4.7")
GENERATOR_MODEL_ID = os.environ.get("EVAL_GENERATOR_MODEL_ID", "glm-4-flash")
GEN_MAX_TOKENS = int(os.environ.get("EVAL_GEN_MAX_TOKENS", "8192"))
JUDGE_MAX_TOKENS = int(os.environ.get("EVAL_JUDGE_MAX_TOKENS", "2048"))

# 裁判模型：RAGAS 用它来打分。
# ⚠️ 默认与被测系统同款，但同款模型给自己打分会有"自我偏好"，分数偏高。
#    真要出结论，最好在 .env 里配一个不同的模型（哪怕是同厂的另一个档位）。
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", MODEL_ID)

# 嵌入模型：必须与检索用的保持一致，否则语义相似度不可比
EMBEDDING_MODEL = os.environ.get("EVAL_EMBEDDING_MODEL", "embedding-3")
EMBEDDING_BATCH_SIZE = 16

# ============================================================
# --- 测试集生成 ---
# ============================================================
TESTSET_SIZE = 30            # 生成多少道题
TESTSET_LANGUAGE = "chinese" # 用于 adapt_prompts，把出题 prompt 适配成中文
NUM_PERSONAS = 3             # RAGAS 会先造几个"提问者人设"，再按人设出题
RANDOM_SEED = int(os.environ.get("EVAL_RANDOM_SEED", "42"))


# ============================================================
# --- 跑图采集（run_rag.py）---
# ============================================================
# 问题被判"不清晰"时，图会停在 request_clarification 等用户补充。
# 批量评测没人在旁边补充，这里决定怎么处理：
AUTO_CLARIFY_ROUNDS = 1      # 自动补充几轮；设为 0 = 直接记为 clarification 跳过
AUTO_CLARIFY_REPLY = "就按问题字面意思回答，直接依据文档内容作答即可。"

# 检索原文的清洗
STRIP_CONTEXT_HEADER = True          # 去掉 "Parent ID: / File Name: / Content:" 这三行头
DROP_CHILD_COVERED_BY_PARENT = True  # 子块内容若已被同批父块包含，就丢掉（避免同一段原文重复计分）
MAX_CONTEXTS_PER_SAMPLE = 20         # 单条样本最多保留多少个原文块（护住裁判模型的 token 开销）


# ============================================================
# --- 评分（score.py）---
# ============================================================
JUDGE_MAX_WORKERS = int(os.environ.get("EVAL_JUDGE_MAX_WORKERS", "1"))
JUDGE_TIMEOUT = 300      # 单个指标调用超时（秒）
JUDGE_MAX_RETRIES = int(os.environ.get("EVAL_JUDGE_MAX_RETRIES", "5"))
JUDGE_MAX_WAIT = int(os.environ.get("EVAL_JUDGE_MAX_WAIT", "120"))
JUDGE_DISABLE_THINKING = os.environ.get("EVAL_JUDGE_DISABLE_THINKING", "true").lower() == "true"
JUDGE_HEARTBEAT_SECONDS = int(os.environ.get("EVAL_JUDGE_HEARTBEAT_SECONDS", "15"))

# 只限制送给裁判的检索上下文，不修改 runs.jsonl 中保存的真实检索结果。
# 按 reranker 顺序保留前面的块，降低裁判延迟和 token 开销。
JUDGE_MAX_CONTEXTS = int(os.environ.get("EVAL_JUDGE_MAX_CONTEXTS", "8"))
JUDGE_MAX_CONTEXT_CHARS = int(os.environ.get("EVAL_JUDGE_MAX_CONTEXT_CHARS", "5000"))

# ResponseRelevancy 默认要求一次生成 3 个候选问题，但部分 OpenAI 兼容接口
# 忽略 n=3 并只返回 1 个。设为 1 可消除警告，并让评分行为与实际接口一致。
ANSWER_RELEVANCY_STRICTNESS = int(os.environ.get("EVAL_ANSWER_RELEVANCY_STRICTNESS", "1"))
