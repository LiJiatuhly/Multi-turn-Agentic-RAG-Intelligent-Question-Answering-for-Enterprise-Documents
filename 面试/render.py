# 把 manual.html 渲染成 PDF（A4，页脚带页码）。
# 用法：python3 render.py   （通常由 build.sh 调用）
# 依赖：pip install playwright  && playwright install chromium
import asyncio, os
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = "file://" + os.path.join(HERE, "manual.html")
OUT  = os.path.join(HERE, "罗氏面试准备_郭一帆.pdf")

# 页脚：左边固定署名，右边“第 X / N 页”
FOOT = (
    '<div style="width:100%;font-size:7pt;color:#8A97A4;'
    'font-family:\'Noto Sans CJK SC\',sans-serif;padding:0 14mm;'
    'display:flex;justify-content:space-between;align-items:center;">'
    '<span>郭一帆 · 罗氏 AI &amp; Data Science Intern 面试准备</span>'
    '<span>第 <span class="pageNumber"></span> / <span class="totalPages"></span> 页</span>'
    '</div>'
)
HEAD = '<div></div>'   # 不要页眉

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--font-render-hinting=none"])
        page = await browser.new_page()
        await page.goto(SRC, wait_until="networkidle")
        await page.wait_for_timeout(600)   # 等 SVG 图片渲染稳定
        await page.pdf(
            path=OUT, format="A4", print_background=True,
            display_header_footer=True, header_template=HEAD, footer_template=FOOT,
            margin={"top": "13mm", "bottom": "15mm", "left": "15mm", "right": "15mm"},
        )
        await browser.close()
    print("已生成:", OUT)

asyncio.run(main())
