from PIL import Image
import numpy as np

img = np.asarray(Image.open(r"E:\RETFound\REFUGE2\Train\mask\0001.bmp").convert("L"))
values, counts = np.unique(img, return_counts=True)

print("灰度值 - 像素数量 - 占比")
for v, c in zip(values, counts):
    ratio = c / img.size * 100
    print(f"{v:3d}    → {c:8d} 像素 ({ratio:.2f}%)")