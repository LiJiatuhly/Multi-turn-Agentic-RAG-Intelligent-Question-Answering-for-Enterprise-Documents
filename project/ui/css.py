# 界面样式(CSS)：亮色主题下的轻量美化。
# 配色（画布/卡片/边框/正文/次要文字/按钮）已全部由主题按 shadcn 规范设定（见 gradio_app.py 的 THEME）。
# 这里只做：隐藏页脚、居中容器、标题栏、圆角。颜色一律用主题变量，不硬写死。
custom_css = """
footer{display:none !important;}
.progress-text{display:none !important;}

.gradio-container{max-width:1040px !important; margin:0 auto !important;}

/* 顶部标题栏：底部一条 slate-200 发丝分隔线 */
.app-header{padding:24px 4px 14px; border-bottom:1px solid var(--border-color-primary); margin-bottom:12px;}
.app-header .title{
    font-size:23px; font-weight:600; letter-spacing:.2px;
    display:flex; align-items:center; gap:10px;
    color:var(--body-text-color);
}
.app-header .pill{
    font-size:12px; font-weight:500; padding:3px 11px; border-radius:999px;
    background:var(--primary-50); color:var(--primary-600);
    border:1px solid var(--primary-200);
}
.app-header .subtitle{
    font-size:13px; margin-top:7px; line-height:1.6;
    color:var(--body-text-color-subdued);
}

/* 聊天窗口和气泡：圆角柔和一点，颜色交给主题 */
.chatbot{border-radius:14px !important;}
.message{border-radius:12px !important; line-height:1.7 !important;}

/* 上传拖放区圆角 */
.file-preview,[data-testid="file-upload"]{border-radius:12px !important;}
"""
