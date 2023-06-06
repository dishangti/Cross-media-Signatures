'''
Make regular images square.
'''
# Run at the same directory level of folder "rect.

import os
from PIL import Image

for i in range(1, 101):   # Volunteer NO
    path = f'./rect/{i}/'
    fileList=os.listdir(path)   # File list

    for file in fileList:
        if ('.png' not in file) and ('.jpg' not in file):   # Obtaining only the image files (skipping any directory configuration files that may be generated on Windows).
            continue
        inp = file
        im = Image.open(path + inp)
        x,y = im.size
        size=max(x,y)
        img=Image.new('RGBA', (size, size), (255,255,255,255))  # Filling a white opaque square background board.
        img.paste(im, ((size - x) // 2, (size - y) // 2))   # Placing the original image in the center.
        if not os.path.exists(f'./square/{i}'):
            os.makedirs(f'./square/{i}')
        img.save(f'./square/{i}/'+file[:-4]+".png", 'PNG')
