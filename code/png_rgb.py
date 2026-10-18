# -*- coding: utf-8 -*-
"""_png_rgb.py — Elsevier 要求 RGB 位图；matplotlib 默认写 RGBA

把 results/*.png 统一转成 RGB（白底合成，透明通道实际全不透明），保留 300 dpi
元数据与像素尺寸不变。每次重绘后、构建 docx 前跑一次。
"""
import glob
import os

from PIL import Image

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

n = 0
for p in sorted(glob.glob(os.path.join(RES, "*.png"))):
    im = Image.open(p)
    if im.mode == "RGB":
        continue
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        out = bg
    else:
        out = im.convert("RGB")
    dpi = im.info.get("dpi", (300, 300))
    out.save(p, dpi=dpi)
    n += 1

print(f"[png] {n} file(s) converted to RGB")
