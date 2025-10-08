import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps


# -------------------
# Utility Functions
# -------------------
def get_gaussian_kernel(size=5, sigma=1):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def gaussian_filter(image, ksize=5, sigma=1):
    gimage = np.array(image)
    kernel = get_gaussian_kernel(ksize, sigma)
    kernel = np.fft.fft2(kernel, image.shape)

    gimage = np.fft.fft2(gimage)
    gimage = gimage * kernel
    return np.fft.ifft2(gimage).real


def convolution(image, kernel):
    gimage = np.array(image)
    h = np.fft.fft2(kernel, image.shape)
    gimage = np.fft.fft2(gimage)
    gimage = gimage * h
    return np.fft.ifft2(gimage).real


def sobel_gradients(img):
    sobelX = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]

    sobelY = [[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]]

    gx = convolution(img, sobelX)
    gy = convolution(img, sobelY)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag / mag.max()) * 255
    return mag, np.arctan2(gy, gx + 1e-9)


def non_max_suppression(mag, grad):
    nms = np.zeros(mag.shape)
    grad = np.rad2deg(grad) + 180
    rows, cols = mag.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            angle = grad[i][j]
            q, r = 255, 255

            if (112.5 <= angle <= 157.5) or (292.5 <= angle <= 337.5):
                q, r = mag[i-1][j-1], mag[i+1][j+1]
            elif (22.5 <= angle <= 67.5) or (202.5 <= angle <= 247.5):
                q, r = mag[i-1][j+1], mag[i+1][j-1]
            elif (67.5 <= angle <= 112.5) or (247.5 <= angle <= 292.5):
                q, r = mag[i-1][j], mag[i+1][j]
            else:
                q, r = mag[i][j-1], mag[i][j+1]

            if mag[i][j] >= q and mag[i][j] >= r:
                nms[i][j] = mag[i][j]

    return nms


def double_threshold(img, low_ratio=0.05, high_ratio=0.15, weak=50, strong=255):
    high = np.max(img) * high_ratio
    low = high * low_ratio
    out = np.zeros(img.shape)
    out[(img > low) & (img <= high)] = weak
    out[img >= high] = strong
    return out


def hysteresis(img, weak=50, strong=255):
    M, N = img.shape
    out = np.copy(img)

    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i, j] == weak:
                if np.any(out[i-1:i+2, j-1:j+2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out


# -------------------
# Streamlit UI
# -------------------
st.header("Canny Edge Detector")
st.write("Upload an image and tune the parameters to see the Canny edge detection pipeline.")

t_sigma = st.number_input("Standard Deviation for Gaussian Filter", 1.0)
low_ratio = st.number_input("Lower Threshold Ratio", 0.05)
high_ratio = st.number_input("Higher Threshold Ratio", 0.15)

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    gray = ImageOps.grayscale(original_image)
    img = np.asarray(gray)

    blurred = gaussian_filter(img, 5, t_sigma)
    mag, grad = sobel_gradients(blurred)
    nms = non_max_suppression(mag, grad)
    dt = double_threshold(nms, low_ratio, high_ratio)
    final = hysteresis(dt)

    st.image(original_image, caption="Original Image")
    st.image(blurred, caption="Gaussian Blur")
    st.image(mag, caption="Sobel Gradients")
    st.image(nms, caption="Non-Max Suppression")
    st.image(dt, caption="Double Thresholding")
    st.image(final, caption="Final Output (Hysteresis)")
