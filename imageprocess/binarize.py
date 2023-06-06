'''
Binarize images.
'''
# Run at the same directory level of folder "sqaure".

import os
import cv2
import numpy as np
from skimage import morphology

size = 224

for i in range(1, 101):     # Volunteer NO 
    path = 'square/'+str(i) + '/'
    if not os.path.exists(f'bin/{i}'):
        os.makedirs(f'bin/{i}')

    for j in range(1, 41):          # Signature NO 
        if os.path.exists(path + str(j)+'.png'):        # Determining the image format.
            file = str(j)+'.png'
        elif os.path.exists(path + str(j)+'.jpg'):
            file = str(j)+'.jpg'
        else:
            continue
        img = cv2.imread(path + file, 0)   # Read image


        if 1 <= j <= 10:  thresh = 100
        elif 11 <= j <= 20: thresh = 100
        elif 21 <= j <= 40: thresh = 200        # etting binarization threshold based on different media characteristics (a few images may require separate adjustment).
        ret, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)     # Binarization: converting the image to binary using a threshold value of thresh. Applying cv2.THRESH_BINARY_INV to invert the result.

        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)     # Resizing the image to a specific square size using the cv2.INTER_AREA compression algorithm.
        
        img = 255 - img    # Reverting the image back to make the text black and the background white after inverting during binarization.

        cv2.imwrite(f'bin/{i}/{file}', img)  # Output image to file