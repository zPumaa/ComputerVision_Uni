import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, maximum_filter 
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

sigma = 0.5
ksize = 5
gaussian_kernel_1d = cv2.getGaussianKernel(ksize, sigma)
gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)

    # Apply the Gaussian kernel to each of the gradient images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Ixx = cv2.filter2D(np.square(Ix), -1, gaussian_kernel_2d)
    # Iyy = cv2.filter2D(np.square(Iy), -1, gaussian_kernel_2d)
    # Ixy = cv2.filter2D(np.multiply(Ix, Iy), -1, gaussian_kernel_2d)


def HarrisPointsDetector(image, blockSize=5, apertureSize=3, k=0.05, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.CV_64F  
    Ix = cv2.Sobel(gray, ddepth, 1, 0, ksize=apertureSize, borderType=cv2.BORDER_REFLECT101)
    Iy = cv2.Sobel(gray, ddepth, 0, 1, ksize=apertureSize, borderType=cv2.BORDER_REFLECT101)

    # Weighting the gradients with a Gaussian filter
    Ixx = gaussian_filter(Ix**2, sigma=0.5)
    Iyy = gaussian_filter(Iy**2, sigma=0.5)
    Ixy = gaussian_filter(Ix*Iy, sigma=0.5)


    # Compute the Harris matrix M for each pixel
    detM = (Ixx * Iyy) - (Ixy ** 2)
    traceM = Ixx + Iyy
    R = detM - (k * (traceM ** 2))

    # Orientation calculation
    orientation = np.arctan2(Iy, Ix)
    orientation = np.rad2deg(orientation)
    orientation = (orientation + 360) % 360  # Convert to range [0, 360)

    # Non-maxima suppression to get keypoints
    local_max = maximum_filter(R, size=(7,7))
    R_max = (R == local_max)
    keypoints = np.argwhere(R_max)
    keypoints_with_scores = [
        cv2.KeyPoint(float(x[1]), float(x[0]), blockSize, angle=float(orientation[x[0], x[1]]), response=float(R[x[0], x[1]]))
        for x in keypoints
    ]

    threshold = threshold * R.max()
    keypoints_final = [kp for kp in keypoints_with_scores if kp.response > threshold]
    return keypoints_final

def featureDescriptor(image, keypoints):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the descriptors with ORB for the keypoints
    _, descriptors = orb.compute(gray, keypoints)

    return descriptors

def ssd_match(descriptors1, descriptors2):
    distances = cdist(descriptors1, descriptors2, 'sqeuclidean')
    matches = []
    for i in range(len(distances)):
        best_match_idx = np.argmin(distances[i])
        matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_match_idx, _distance=distances[i][best_match_idx]))
    return matches

def ratio_match(descriptors1, descriptors2, ratio=0.7):
    distances = cdist(descriptors1, descriptors2, 'sqeuclidean')
    matches = []
    for i in range(len(distances)):
        ordered_distance = np.argsort(distances[i])
        if len(ordered_distance) > 1:
            if distances[i][ordered_distance[0]] < ratio * distances[i][ordered_distance[1]]:
                matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=ordered_distance[0], _distance=distances[i][ordered_distance[0]]))
    return matches


defaultBernie = cv2.imread('bernieSanders.jpg')
brightBernie = cv2.imread('darkerBernie.jpg')
height1 = defaultBernie.shape[0]
width1 = defaultBernie.shape[1]

height2 = brightBernie.shape[0]
width2 = brightBernie.shape[1]

# Calculate the scaling factor
scaling_factor = height1 / float(height2)

# Resize the second image
brightBernie = cv2.resize(brightBernie, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
start_time_harris = time.time()
keypoints = HarrisPointsDetector(defaultBernie)
end_time_harris = time.time()
harris_duration = end_time_harris - start_time_harris
print(f"HarrisPointsDetector took {harris_duration:.4f} seconds.")
descriptors = featureDescriptor(defaultBernie, keypoints)


bright_kp = HarrisPointsDetector(brightBernie)
bright_desc = featureDescriptor(brightBernie, bright_kp)

# Detect keypoints with built-in ORB
orb = cv2.ORB_create()
start_time_fast = time.time()
orb_keypoints = orb.detect(defaultBernie, None)
end_time_fast = time.time()
fast_duration = end_time_fast - start_time_fast
print(f"Fast took {fast_duration:.4f} seconds.")
orb_descriptors = featureDescriptor(defaultBernie, orb_keypoints)

brightorb_keypoints = orb.detect(brightBernie, None)
brightorb_descriptors = featureDescriptor(brightBernie, brightorb_keypoints)

print(f"Harris detected {len(keypoints)} points.")
# print(f"ORB detected {len(orb_keypoints)} points.")

img2 = cv2.drawKeypoints(defaultBernie, keypoints, None, color=(17,255,41), flags=0)
# img3 = cv2.drawKeypoints(defaultBernie, orb_keypoints, None, color=(17,255,41), flags=0)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Display img2 in the first subplot
axes[0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
axes[0].axis('off')  

# Display img3 in the second subplot
# axes[1].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
axes[1].axis('off')  
plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()


# Use the match functions
ssd_matches = ssd_match(orb_descriptors, brightorb_descriptors)
ratio_matches = ratio_match(orb_descriptors, brightorb_descriptors, ratio=0.4)
matches = bf.match(orb_descriptors, brightorb_descriptors)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches using cv2.drawMatches
matched_image_ssd = cv2.drawMatches(defaultBernie, orb_keypoints, brightBernie, brightorb_keypoints, ssd_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
matched_image_ratio = cv2.drawMatches(defaultBernie, orb_keypoints, brightBernie, brightorb_keypoints, ratio_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches = cv2.drawMatches(defaultBernie, orb_keypoints, brightBernie, brightorb_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('noisy_ssd.jpg', matched_image_ssd)
cv2.imwrite('darker_ratio_0.4.jpg', matched_image_ratio)
# cv2.imwrite('img_matches.jpg', img_matches)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Display img2 in the first subplot
axes[0].imshow(cv2.cvtColor(matched_image_ssd, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
axes[0].axis('off')  

# Display img3 in the second subplot
axes[1].imshow(cv2.cvtColor(matched_image_ratio, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
axes[1].axis('off')  
plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
# Show the matched images
# cv2.imshow("SSD Matches", matched_image_ssd)
# cv2.imshow("Ratio Test Matches", matched_image_ratio)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




threshold_values = np.linspace(0.01, 0.1, 10)
keypoint_counts = []

for threshold in threshold_values:
    keypoints = HarrisPointsDetector(defaultBernie, threshold=threshold)
    keypoint_count = len(keypoints)
    keypoint_counts.append(keypoint_count)

plt.figure(figsize=(10, 6))
plt.plot(threshold_values, keypoint_counts, marker='o')
plt.title('Number of Keypoints vs Threshold Values')
plt.xlabel('Threshold Value')
plt.ylabel('Number of Keypoints')
# Set the y-axis limits to expected range of keypoint counts
plt.ylim([min(keypoint_counts) - 10, max(keypoint_counts) + 10])
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()

# print(keypoints[0].pt, keypoints[0].size, keypoints[0].angle)
