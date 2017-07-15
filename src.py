# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

TEST_DIR = "test_images"
files = os.listdir(TEST_DIR)

# vertices = np.array(
# [[(0,imshape[0]),
#   (450, 290),
#   (490, 290),
#   (imshape[1],imshape[0])]], dtype=np.int32)

# I chose parameters for my Hough space grid to be a rho of 2 pixels and theta of 1 degree (pi/180 radians).
# I chose a threshold of 15, meaning at least 15 points in image space need to be associated with each line segment.
# I imposed a min_line_length of 40 pixels, and max_line_gap of 20 pixels.

# Read in and grayscale the image
image = mpimg.imread(os.path.join(TEST_DIR, files[0]))
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 


import utils
imshape = combo.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

combo = utils.region_of_interest(combo, vertices)

plt.imshow(image)
plt.show()
plt.imshow(combo)
plt.show()

print("done")

