import cv2
import numpy as np

def process_image(image_path):
    # 📥 Read the image in grayscale using '0' (shortcut for cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path, 0)

    # 1️⃣ Histogram Equalization - improve contrast
    equalized = cv2.equalizeHist(img)

    # 2️⃣ Thresholding - convert to pure black and white
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 3️⃣ Edge Detection - detect edges using Canny
    edges = cv2.Canny(img, 100, 200)

    # 4️⃣ Data Augmentation - flip the image horizontally
    flipped = cv2.flip(img, 1)

    # 5️⃣ Morphological Operation - closing to remove small black holes
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 🖼️ Display all images
    cv2.imshow('Original', img)
    cv2.imshow('Equalized', equalized)
    cv2.imshow('Thresholded', thresholded)
    cv2.imshow('Edges', edges)
    cv2.imshow('Flipped', flipped)
    cv2.imshow('Morphed (Close)', morphed)

    # ⏳ Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 🔁 Run it with your image path
process_image('/home/lab705/Downloads/1.jpeg')
