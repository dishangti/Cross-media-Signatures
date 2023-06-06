'''
Binarize and then skeletonize images.
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

        # skel = np.zeros(img.shape, np.uint8)                            # Skeletonization: morphological iteration (the result may not be optimal).
        # element = c,v2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        # while True:
        #     open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #     temp = cv2.subtract(img, open)
        #     eroded = cv2.erode(img, element)
        #     skel = cv2.bitwise_or(skel,temp)
        #     img = eroded.copy()
        #     if cv2.countNonZero(img)==0:
        #         break
        
        for m in range(img.shape[0]):           # Skeletonization: using the built-in skeletonization algorithm in skimage.
            for n in range(img.shape[1]):
                if img[m][n] == 255:
                    img[m][n] = 1               # Please note that the library requires the binary image to have pixel values of 0 or 1, instead of 0 or 255.
        skeleton = morphology.skeletonize(img)
        img = skeleton.astype(np.uint8)*255     # Convert the binary image back to have pixel values of 0 or 255.
        img = 255 - img    # Reverting the image back to make the text black and the background white after inverting during binarization.

        cv2.imwrite(f'skel/{i}/{file}', img)  # Output image to file