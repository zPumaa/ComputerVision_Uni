import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt



# ================================================
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================


# ================================================
def plot(disparity, f, cx, cy, baseline, doffs):
    h, w = disparity.shape

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate Z, X, and Y for each pixel
    Z = (baseline * f) / (disparity.astype(np.float32) + doffs)
    X = (x - cx) * Z / f
    Y = (y - cy) * Z / f

    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Calculate the maximum value of Z and set the threshold to 98% of this value
    max_Z_value = np.max(Z)
    Z_threshold = 0.98 * max_Z_value

    # Filter out points where Z is above the threshold
    mask = Z < Z_threshold
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    Z_filtered = Z[mask]

    fig = plt.figure(figsize=(15, 5))

    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_filtered, Y_filtered, Z_filtered, c='green', marker='.')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-2000, 2000])
    ax1.set_ylim([-2000, 2000])
    ax1.set_zlim([8000, Z_threshold])
    ax1.title.set_text('3D View')

    # Top view (x,z)
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_filtered, Z_filtered, c='red', marker='.')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_xlim([-2000, 2000])
    ax2.set_ylim([8500, Z_threshold])
    ax2.title.set_text('Top View (X,Z)')

    # Side view (y,z)
    ax3 = fig.add_subplot(133)
    ax3.scatter(Y_filtered, Z_filtered, c='blue', marker='.')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_xlim([-2000, 2000])
    ax3.set_ylim([8500, Z_threshold])
    ax3.title.set_text('Side View (Y,Z)')

    plt.tight_layout()
    plt.show()

def calc_focal_length(f_px, sw ,iw):
    f_mm = f_px * (sw / iw)
    return f_mm

focal_length = calc_focal_length(5806.559, 22.2, 3088)
print(focal_length)

# Global variables to store the trackbar values
numDisparities = 64 
blockSize = 5  

def on_trackbar_disp(val):
    # global numDisparities
    if val % 16 == 0 and val != 0:
        numDisparities = val
    disparity = getDisparityMap(edgesL, edgesR, numDisparities, blockSize)
    # disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)


def on_trackbar_block(val):
    #global blockSize
    blockSize = val 
    blockSize = max(5, blockSize + (blockSize % 2 == 0))
    disparity = getDisparityMap(edgesL, edgesR, numDisparities, blockSize)
    # disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)


f_original = 5806.559  # Focal length in pixels for the original image size
cx_original = 1429.219  # The x-coordinate of the principal point for the original image size
cy_original = 993.403  # The y-coordinate of the principal point for the original image size
doffs_original = 114.291  # The x-difference of the principal points
baseline = 174.019  # The camera baseline in millimeters

# Original and resized image dimensions
width_original, height_original = 2960, 2016
width_resized, height_resized = 740, 505

scale_factor = width_resized / width_original

f = scale_factor * f_original
cx0 = scale_factor * cx_original
cy = scale_factor * cy_original
doffs = scale_factor * doffs_original

if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()
    edgesL = cv2.Canny(imgL, 50, 150)

    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()
    edgesR = cv2.Canny(imgR, 50, 150)


    # Initialize the disparity settings
    numDisparities = 64
    blockSize = 5
    
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('numDisparities', 'Disparity', numDisparities, 256, on_trackbar_disp)
    cv2.createTrackbar('blockSize', 'Disparity', blockSize, 50, on_trackbar_block)

    # Recalculate the disparity map with the new parameters
    disparity = getDisparityMap(edgesL, edgesR, numDisparities, blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)
    plot(disparityImg, f, cx0, cy, baseline, doffs)

    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
