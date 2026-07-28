#!/usr/bin/env bash
# 一键重建 PDF：先把两张流程图 .dot 渲染成 .svg，再把 manual.html 渲染成 PDF。
# 用法：改完 manual.html / fc1.dot / fc2.dot 后，在本目录运行：  bash build.sh
set -e
cd "$(dirname "$0")"

echo "[1/2] 流程图 .dot -> .svg ..."
dot -Tsvg fc1.dot -o fc1.svg
dot -Tsvg fc2.dot -o fc2.svg

echo "[2/2] manual.html -> PDF ..."
python3 render.py

echo "完成 ✅  ->  罗氏面试准备_郭一帆.pdf"
