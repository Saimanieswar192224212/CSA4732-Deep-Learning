import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread(r"C:\Users\Madhu\OneDrive\Pictures\Linkedin_cover_photo_1.png")
if img is None:
    raise FileNotFoundError("Image not found at the specified path")

# Convert BGR to RGB for displaying with matplotlib
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define kernel for morphological operations
kernel = np.ones((2, 2), np.uint8)

# Morphological closing operation
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Dilate the image
sure_bg = cv2.dilate(closing, kernel, iterations=2)

# Plot the results
plt.subplot(211), plt.imshow(closing, 'gray')
plt.title("MorphologyEx: Closing (2x2 Kernel)"), plt.xticks([]), plt.yticks([])

plt.subplot(212), plt.imshow(sure_bg, 'gray')
plt.title("Dilation"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

# Save the dilation result
cv2.imwrite(r'C:\Users\Madhu\OneDrive\Pictures\dilation.png', sure_bg)
