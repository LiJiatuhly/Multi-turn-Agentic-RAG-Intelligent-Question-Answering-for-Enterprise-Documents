# 程序启动入口：加载 .env、创建并启动 Gradio 界面。
# 只有这一个文件需要直接运行：python project/app.py

import sys
import os
import logging

# 把 project/ 目录加入 Python 搜索路径，
# 这样 project/ 下的模块可以直接 import（不用写 project.xxx）
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
# 加载 project/.env 文件里的环境变量（API_KEY 等）
# 必须在其他 import 之前执行，因为 config.py 在 import 时就读 os.environ
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 屏蔽 OpenTelemetry 的 "Failed to detach context" 警告（由生成器/上下文交互引起）
# 不影响功能与追踪；这是一个已知的第三方库问题
class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

from ui.gradio_app import create_gradio_ui, THEME, FORCE_LIGHT_JS
from ui.css import custom_css

# if：只有"直接运行本文件"时才启动界面（被别的文件 import 时不会执行这里）
if __name__ == "__main__":
    print("\n正在创建 RAG 助手...")
    demo = create_gradio_ui()      # 初始化 RAGSystem + 编译 Agent 图（这里会连一次智谱）
    print("\n正在启动 RAG 助手...")
    # Gradio 6：主题、CSS、强制亮色的 JS 都在 launch() 里传（默认 http://127.0.0.1:7860）
    demo.launch(theme=THEME, css=custom_css, js=FORCE_LIGHT_JS)
