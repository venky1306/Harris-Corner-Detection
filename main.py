import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Read image
    img = cv2.imread("img.png")

    # Resize image with 300x300
    img = cv2.resize(img, (300, 300))


    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Gradient images in x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    #Computing Product of x and y gradients at every pixel
    sobelxy = sobelx * sobely
    sobelxx = sobelx * sobelx
    sobelyy = sobely * sobely

    #Computing the sum of the products of x and y gradients at every pixel using Gaussian filter
    sobelxy = cv2.GaussianBlur(sobelxy, (5, 5), 0)
    sobelxx = cv2.GaussianBlur(sobelxx, (5, 5), 0)
    sobelyy = cv2.GaussianBlur(sobelyy, (5, 5), 0)

    #Compute the determinant and the trace of the M matrix
    det_M = sobelxx * sobelyy - sobelxy * sobelxy
    trace_M = sobelxx + sobelyy

    #Compute the Harris corner score
    R = det_M - 0.04 * trace_M * trace_M

    #Thresholding the score
    R[R < 0.01 * R.max()] = 0

    #Non-maximum suppression
    height, width = R.shape
    mask = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                continue
            if R[i][j] > max(R[i-1][j-1], R[i-1][j], R[i-1][j+1], R[i][j-1], R[i][j+1], R[i+1][j-1], R[i+1][j], R[i+1][j+1]):
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    
    #Displaying the original image and the detected corners side by side
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(mask, cmap='gray')
    plt.title('Corners'), plt.xticks([]), plt.yticks([])
    plt.show()
    




