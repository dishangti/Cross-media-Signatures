'''
Extract signatures from photos and make them square images.
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
    im = im.convert('L')        # Convert the image to grayscale for easier processing.
    x,y = im.size
    for w in range(x):
        for h in range(y):
            pix = im.getpixel((w, h))        # Accessing the pixels.
            if pix >= 50:
                im.putpixel((w, h), (0))       # The threshold should be able to separate the background and the strokes. Fill the strokes with black (0) and the background with white (1).
            else:
                im.putpixel((w, h), (255))
    size=max(x,y)
    img=Image.new('RGBA', (size, size), (255,255,255,255))  # Filling a white opaque square background.
    img.paste(im, ((size - x) // 2, (size - y) // 2))   # Placing the original image in the center.
    if not os.path.exists(f'./photo'):
        os.makedirs(f'./photo')
    img.save(f'./photo/{i}.png','PNG')
