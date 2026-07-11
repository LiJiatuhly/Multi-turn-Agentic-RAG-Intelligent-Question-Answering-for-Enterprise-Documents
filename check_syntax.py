# 语法自检脚本：逐个检查 project/ 下所有 .py 文件有没有语法错误。
# 用法（在项目根目录、且已激活 venv 后）：  python check_syntax.py
import py_compile, pathlib, sys

root = pathlib.Path(__file__).parent / "project"
files = sorted(root.rglob("*.py"))
ok = 0
bad = 0
print(f"开始逐个语法自检，共 {len(files)} 个文件：\n")
for f in files:
    rel = f.relative_to(pathlib.Path(__file__).parent)
    try:
        py_compile.compile(str(f), doraise=True)
        print(f"  [OK]    {rel}")
        ok += 1
    except py_compile.PyCompileError as e:
        print(f"  [错误]  {rel}")
        print(f"          {e.msg.strip()}")
        bad += 1

print(f"\n检查完成：通过 {ok} 个，失败 {bad} 个。")
if bad == 0:
    print("✅ 所有文件语法正常，可以启动了：python project/app.py")
    sys.exit(0)
else:
    print("❌ 有文件语法有误，请按上面提示修复后再启动。")
    sys.exit(1)
