#!/usr/bin/env python3
"""Create a smaller test image for faster ablation testing."""
import cv2
img = cv2.imread('input/REAL_WORLD_PICTURE.jpg')
img_small = cv2.resize(img, (1600, 740))
cv2.imwrite('input/test_small.jpg', img_small)
print(f'Created 1600x740 test image from {img.shape[1]}x{img.shape[0]}')
