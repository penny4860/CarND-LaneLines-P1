# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

from utils import grayscale, canny, gaussian_blur, region_of_interest, hough_lines


TEST_DIR = "test_images"
files = os.listdir(TEST_DIR)

# Read in and grayscale the image
image = mpimg.imread(os.path.join(TEST_DIR, files[0]))
imshape = image.shape
xlength = imshape[1]
ylength = imshape[0]
gray = grayscale(image)

# Define a kernel size and apply Gaussian smoothing
blur_gray = gaussian_blur(gray, kernel_size=5)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = canny(blur_gray, low_threshold, high_threshold)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20

# Run Hough on edge detected image
line_image = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)

vertices = np.array([[(0, ylength),
                      (xlength/2-ylength/10, ylength/2),
                      (xlength/2+ylength/10, ylength/2),
                      (xlength, ylength)]], dtype=np.int32)

combo = region_of_interest(line_image, vertices)
 
plt.imshow(image)
plt.show()
plt.imshow(combo)
plt.show()


