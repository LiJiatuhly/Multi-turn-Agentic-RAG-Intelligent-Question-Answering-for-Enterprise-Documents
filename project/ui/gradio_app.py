# Gradio 界面：文档上传标签页 + 对话标签页（含深色主题与顶部标题栏）。
# C 级：知道"哪个按钮触发哪个函数"就够了，不需要逐行背。

import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem
from ui.css import custom_css
import os

# 机器人头像图片的路径（放在 project/assets/ 目录下）
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

# 主题：参考 GitHub 顶流设计系统 shadcn/ui 的亮色规范落到 Gradio 主题变量上——
#   近白画布 + 纯白卡片 + slate-200 发丝边框 + slate-900 正文 + slate-500 次要文字，
#   强调色(indigo)只用在主按钮等高信号处。圆角统一、字体用系统字体(避开谷歌字体在国内被墙)。
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=["system-ui", "-apple-system", "Segoe UI", "Microsoft YaHei", "PingFang SC", "sans-serif"],
).set(
    body_background_fill="#f8fafc",                 # slate-50：柔和的近白画布
    background_fill_primary="#ffffff",              # 纯白卡片
    background_fill_secondary="#f8fafc",
    border_color_primary="#e2e8f0",                 # slate-200：极浅的发丝边框
    body_text_color="#0f172a",                      # slate-900：正文深色，对比清晰
    body_text_color_subdued="#64748b",              # slate-500：次要/说明文字
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_label_text_color="#334155",               # slate-700：小标题
    input_background_fill="#ffffff",
    input_border_color="#e2e8f0",
    button_primary_background_fill="#4f46e5",        # indigo-600：主按钮（强调色只用在这类地方）
    button_primary_background_fill_hover="#4338ca",  # indigo-700
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",      # 次按钮：白底 + 发丝边框（中性）
    button_secondary_border_color="#e2e8f0",
    button_secondary_text_color="#0f172a",
)

# 顶部标题栏（自定义 HTML，样式在 css.py 的 .app-header 里）
HEADER_HTML = """
<div class="app-header">
  <div class="title">📚 智能体 RAG 问答助手 <span class="pill">智谱 GLM</span></div>
  <div class="subtitle">上传中文 PDF，我会自己检索、思考、调用工具，用中文回答并标注来源。</div>
</div>
"""

# 强制亮色模式：这段 JS 在页面加载时执行，如果 URL 没带 ?__theme=light 就重载一次带上它。
# 这样即使你的浏览器/系统是深色模式，Gradio 也统一用它自带的亮色配色（白底卡片、深色文字），
# 所有组件、折叠面板、占位符都一致，不需要我们用 CSS 手动刷颜色（那样容易出现颜色对不上的问题）。
FORCE_LIGHT_JS = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


def create_gradio_ui():
    """C 级 创建并返回 Gradio Blocks 对象（调用 .launch() 才真正启动）。"""
    rag_system = RAGSystem()
    rag_system.initialize()

    doc_manager    = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)

    def format_file_list():
        files = doc_manager.get_markdown_files()
        # if：知识库为空 → 显示占位提示
        if not files:
            return "📭 知识库里还没有任何文档"
        return "\n".join([f"📄 {f}" for f in files])

    def upload_handler(files, progress=gr.Progress()):
        # if：没选任何文件 → 直接刷新列表返回
        if not files:
            return None, format_file_list()
        added, skipped = doc_manager.add_documents(
            files,
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        gr.Info(f"已添加 {added} 个 · 跳过 {skipped} 个")
        return None, format_file_list()

    def clear_handler():
        try:
            doc_manager.clear_all()
            gr.Info("已清空全部文档")
        except Exception as exc:
            gr.Error(f"清空文档失败: {exc}")
        return format_file_list()

    def chat_handler(msg, hist):
        for chunk in chat_interface.chat(msg, hist):
            yield chunk

    def clear_chat_handler():
        chat_interface.clear_session()

    # 注意：Gradio 6 里 theme/css/js 要传给 launch()（见 app.py），不再放在 Blocks() 里
    with gr.Blocks(title="智能体 RAG 中文问答助手") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Tab("文档管理", elem_id="doc-management-tab"):
            gr.Markdown("### 添加新文档")
            gr.Markdown("上传 PDF 或 Markdown 文件。已存在的同名文档会被跳过；如需重新索引，请先点「清空全部」。")

            files_input = gr.File(
                label="把 PDF 或 Markdown 文件拖到这里",
                file_count="multiple",
                type="filepath",
                height=190,
                show_label=False
            )
            add_btn = gr.Button("＋ 添加文档", variant="primary", size="lg")

            gr.Markdown("### 知识库中的当前文档")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines=7,
                max_lines=12,
                elem_id="file-list-box",
                show_label=False
            )

            with gr.Row():
                refresh_btn = gr.Button("↻ 刷新", size="md", variant="secondary")
                clear_btn   = gr.Button("清空全部", variant="stop", size="md")

            add_btn.click(upload_handler, [files_input], [files_input, file_list], show_progress="corner")
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)

        with gr.Tab("对话"):
            chatbot = gr.Chatbot(
                height=680,
                placeholder="<strong>有什么想问的尽管问！</strong><br><em>我会自己检索、思考、调用工具，尽力给你最好的答案 :)</em>",
                show_label=False,
                avatar_images=(None, os.path.join(ASSETS_DIR, "chatbot_avatar.png")),
                layout="bubble"
            )
            chatbot.clear(clear_chat_handler)
            gr.ChatInterface(fn=chat_handler, chatbot=chatbot)

    return demo
