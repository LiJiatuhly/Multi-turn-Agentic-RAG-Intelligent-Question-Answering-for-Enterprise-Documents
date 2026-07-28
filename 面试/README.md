# 面试手册 · PDF 生成源码

改内容、重出 PDF，不用重新生成一份。

## 文件说明
| 文件 | 作用 | 你要改的 |
|---|---|---|
| `manual.html` | **手册全部内容 + 样式**（正文、题库、表格、话术都在这里） | ✅ 改文字主要动这个 |
| `fc1.dot` | 图1 外层主图（Graphviz 源码） | 只有改流程图才动 |
| `fc2.dot` | 图2 内层 Agent 子图（Graphviz 源码） | 只有改流程图才动 |
| `render.py` | HTML → PDF（A4、页脚页码） | 一般不用改 |
| `build.sh` | 一键：先渲染两张图，再出 PDF | 直接运行 |
| `fc1.svg` `fc2.svg` | 图的成品（`manual.html` 引用它们） | 由 `build.sh` 自动重生成 |

## 怎么用（三步）
1. 用编辑器改 `manual.html`（找到要改的段落直接编辑；结构是 `<div class="q">…</div>` 一题一块）。
2. 终端进入本目录，运行：
   ```bash
   bash build.sh
   ```
3. 同目录生成 `罗氏面试准备_郭一帆.pdf`。

> 只改文字、没动流程图，也可以只跑 `python3 render.py`（更快，跳过图渲染）。

## 环境依赖（第一次装一次）
```bash
pip install playwright
playwright install chromium
# Graphviz（渲染流程图用）：
#   Ubuntu/Debian:  sudo apt install graphviz
#   macOS:          brew install graphviz
#   Windows:        https://graphviz.org/download/  （装完确保 dot 在 PATH）
```
还需要中文字体 **Noto Sans CJK SC / Noto Serif CJK SC**（大多数系统自带；没有就装 `fonts-noto-cjk`），否则 PDF 里中文会缺字。

## 常见改动位置（在 manual.html 里搜关键词）
- 改流程图文字/连线 → 改 `fc1.dot` / `fc2.dot`，跑 `build.sh`。
- 改真实参数 → 搜 `CHILD_CHUNK_SIZE`、`MAX_ITERATIONS` 等。
- 改 SQL 话术 → 搜 `按你的真实进度校准`。
- 改反问清单 → 搜 `反问环节`。
- 改页脚署名 → 改 `render.py` 里的 `FOOT`。
- 换主色 → 改 `manual.html` 顶部 `:root{ --navy … --blue … }`。
