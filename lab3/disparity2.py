import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt



# ================================================
def getDisparityMap(imL, imR, numDisparities, blockSize):
    print("BlockSize:", blockSize)
    print("NumDisparities:", numDisparities)
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image

def on_trackbar_disp(val):
    # global numDisparities
    # Check if the trackbar refers to numDisparities or blockSize
    if val % 16 == 0 and val != 0:
        numDisparities = val

    # Recalculate the disparity map with the new parameters
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    # disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)


def on_trackbar_block(val):
    # global blockSize

    blockSize = val 

    # Make sure the block size is odd and greater than or equal to 5
    blockSize = max(5, blockSize + (blockSize % 2 == 0))

    # Recalculate the disparity map with the new parameters
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    # disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.imshow('Disparity', disparityImg)

def on_trackbar_k(val):
    # global k

    k = val/100

    # Recalculate the disparity map with the new parameters
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    depth_map = cv2.divide(1.0, cv2.add(disparity, k))
    disparityImg = np.interp(depth_map, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.imshow('Disparity', disparityImg)

def process_background(image, depth_map, k, blur=False, grayscale=False):
    # Threshold the depth map to create a mask for the foreground
    _, foreground_mask = cv2.threshold(depth_map, k, 255, cv2.THRESH_BINARY)
    foreground_mask = foreground_mask.astype(np.uint8)  
    foreground_mask_3c = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)
    foreground = cv2.bitwise_and(image, foreground_mask_3c)

    background_mask = cv2.bitwise_not(foreground_mask)
    background_mask_3c = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    background = cv2.bitwise_and(image, background_mask_3c)

    if blur:
        foreground = cv2.GaussianBlur(foreground, (21, 21), 0)
    if grayscale:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)

    combined_image = cv2.add(foreground, background)

    return combined_image
    

if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'girlR.png'
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    numDisparities = 64
    blockSize = 45
    k2 = 10
    k = 0.99
        
    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('numDisparities', 'Disparity', numDisparities, 256, on_trackbar_disp)
    cv2.createTrackbar('blockSize', 'Disparity', blockSize, 50, on_trackbar_block)
    cv2.createTrackbar('k', 'Disparity', k2, 100, on_trackbar_k)

    disparity = getDisparityMap(imgL, imgR, numDisparities=numDisparities, blockSize=blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    depth_map = cv2.divide(1.0, cv2.add(disparityImg, k))

    new_image = process_background(image, depth_map, k, blur=False, grayscale=True)
    cv2.imshow('Image', new_image)

    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    # disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    