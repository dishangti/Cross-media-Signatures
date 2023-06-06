'''
Extract red signatures and make them square images.
'''
# Run in the folder where signatures locate.

import os
from PIL import Image

for i in range(1, 41):
    if os.path.exists(f'./{i}.jpg'):
        path = f'./{i}.jpg'
    elif os.path.exists(f'./{i}.png'):
        path = f'./{i}.png'
    else: continue
    im = Image.open(path)
    x,y = im.size
    for w in range(x):
        for h in range(y):
            pix = im.getpixel((w, h))
            if (pix[0] >= 20 and pix[1] <= 200 and pix[2] <= 200) or (pix[0] >= 100 and pix[1] <= 230 and pix[2] <= 230):       # 判断红色笔迹
                im.putpixel((w, h), (0,0,0, *pix[-1:]))     # Turning red strokes into black.
    size=max(x,y)
    img=Image.new('RGBA', (size, size), (255,255,255,255))  # Filling a white opaque square background.
    img.paste(im, ((size - x) // 2, (size - y) // 2))   # Placing the original image in the center.
    if not os.path.exists(f'./square/{i}'):
        os.makedirs(f'./square/{i}')
    img.save(f'./red/{i}.png','PNG')