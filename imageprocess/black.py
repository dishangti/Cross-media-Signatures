'''
Extract signatures with black background and make them square images.
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
    im = im.convert('L')    # Converting the image to grayscale for easier processing (merging the RGB channels into a single channel representing brightness).
    x,y = im.size
    for w in range(x):
        for h in range(y):
            pix = im.getpixel((w, h))
            if pix >= 50:
                im.putpixel((w, h), (0))    # Since the background is black and the strokes are white, we need to invert the colors. The bright areas will be changed to black, and the dark areas will be changed to white.
            else:
                im.putpixel((w, h), (255))
    size=max(x,y)
    img=Image.new('RGBA', (size, size), (255,255,255,255))  # Fill a white opaque square background.
    img.paste(im, ((size - x) // 2, (size - y) // 2))   # Place the original image in the center.
    if not os.path.exists(f'./black'):
        os.makedirs(f'./black')
    img.save(f'./black/{i}.png','PNG')
